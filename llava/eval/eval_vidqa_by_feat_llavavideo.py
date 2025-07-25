import gc
import os
import sys
import copy
import json
import math
import time
import pickle
import argparse
import random

import torch
import numpy as np
import pandas as pd
import transformers
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.nn\.modules\.module"
)

## Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
## Add the project directory to sys.path
sys.path.append(project_dir)

from avg_runtime import measure_runtime
from avg_prompt_stat import measure_prompt_stat

from llava.model.framefusion.interface import apply_framefusion
from llava.model.framefusion.models.qwen2.modeling_qwen2_baseline import replace_Qwen2_fastv, replace_Qwen2_merging, replace_Qwen2_streamingllm

from llava import conversation as conversation_lib
from llava.conversation import SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import KeywordsStoppingCriteria
from llava.train.train import ModelArguments, DataArguments, TrainingArguments
from llava.eval.eval_utils import set_cuda_deterministic, parse_sa_cfg, get_data_path, get_evaluator, EvalArguments
from llava.eval.video_dataset import VidQA_Loader_Feature

from token_merging_monkey_patch.dycoke_attn_monkey_patch import Qwen2FlashAttn_with_eager_mode
from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2


## set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_llava_video_model(model_args, data_args, eval_args):
    model_base = model_args.model_base
    model_path = model_args.model_name_or_path
    model_name = "llava_qwen"
    conv_name = "qwen_2"
    ckpt_name = model_path.split("/")[1].lower()
    max_num_frames = data_args.frames_upbound
    tgt_video_fps = data_args.tgt_video_fps
    repeat_idx = eval_args.repeat_idx
    
    dataset_name = eval_args.dataset_name
    anno_filepath, video_root, dataset_dir = get_data_path(dataset_name)

    ## setup dataset root dir
    data_root = ckpt_name if model_base is None else model_base.split("/")[1].lower()
    data_root = os.path.join(f"datasets/{dataset_dir}/preprocess_data", data_root, f"F-{max_num_frames}_fps-{tgt_video_fps}")

    ## set output_dir
    output_dir = parse_sa_cfg(model_args)
    if model_args.mm_newline_position != "grid":
        output_dir = f"{output_dir}_mm-newline-{model_args.mm_newline_position}"
    if repeat_idx > 0:
        output_dir = f"outputs_{dataset_name}_repeat-{repeat_idx}/{output_dir}"
    else:
        output_dir = f"outputs_{dataset_name}/{output_dir}"
    output_dir = os.path.join("outputs_reproduce", output_dir) if eval_args.reproduce else os.path.join("outputs", output_dir)
    # output_dir = os.path.join("outputs_debug", output_dir) if eval_args.reproduce else os.path.join("outputs", output_dir) # use it for debugging time
    
    ## set rope_scaling_factor and attn_mode
    overwrite_config = None
    rope_scaling_factor = model_args.rope_scaling_factor if model_args.rope_scaling_factor is not None else 1.0
    if rope_scaling_factor > 1:
        '''
        256: x2
        320, 384: x3
        512: x4
        '''
        overwrite_config = {}
        overwrite_config["max_position_embeddings"] = int(32768 * rope_scaling_factor)
        overwrite_config["tokenizer_model_max_length"] = int(32768 * rope_scaling_factor)

    attn_implementation = "flash_attention_2"
    if model_args.attn_implementation != "flash_attention_2":
        attn_implementation = model_args.attn_implementation
    
    ## load pretrained weight
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, torch_dtype="bfloat16", overwrite_config=overwrite_config, device_map="auto", attn_implementation=attn_implementation)
    conv_tmpl = conversation_lib.conv_templates[conv_name]
    conversation_lib.default_conversation = conv_tmpl
    tokenizer.conv_tmpl = conv_tmpl

    ## apply monkey patch
    if model_args.sa_pattern == "framefusion":
        # llava-based: cost = 0.3, similarity_lower_bound = 0.6, ratio_lower_bound = 0.1
        model = apply_framefusion(model, cost=model_args.sa_framefusion_cost, similarity_lower_bound=0.6, ratio_lower_bound=0.1)
    elif model_args.sa_pattern == "fastv":
        replace_Qwen2_fastv(model, fastv_k=model_args.sa_start_layer_idx, fastv_r=model_args.sa_fastv_evict_ratio)
    elif model_args.sa_pattern == "framefusion-merge":
        sparsity = [0.0] * 28
        sparsity[model_args.sa_start_layer_idx] = model_args.sa_prune_ratio
        replace_Qwen2_merging(model, sparsity=sparsity)
    elif model_args.sa_pattern == "streamingllm":
        replace_Qwen2_streamingllm(model, init_num=model_args.sa_asa_n_init, length_rate=model_args.sa_asa_n_ratio)
    elif model_args.sa_pattern == "dycoke":
        for name, module in model.named_modules():
            for child_name, child in module.named_children():
                if isinstance(child, Qwen2FlashAttention2):
                    # Replace the attention block
                    new_attn = Qwen2FlashAttn_with_eager_mode(child.config, layer_idx=child.layer_idx)
                    new_attn.to(next(child.parameters()).dtype) 
                    new_attn.to(next(child.parameters()).device) 
                    new_attn.load_state_dict(child.state_dict())
                    setattr(module, child_name, new_attn)

    ## set config
    data_args.image_processor = image_processor
    data_args.add_time_instruction = model.config.add_time_instruction if hasattr(model.config, "add_time_instruction") else False
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False
    
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.max_batch_size = 256
    model.config.mm_spatial_pool_mode = "bilinear"
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride if model_args.mm_spatial_pool_stride is not None else 2
    model.config.slow_mm_spatial_pool_stride = model_args.slow_mm_spatial_pool_stride
    model.config.slow_path_stride = model_args.slow_path_stride
    model.config.slow_fast_path_flag = model_args.slow_fast_path_flag

    ## create answer file
    pool_size = model.config.mm_spatial_pool_stride
    pred_filepath = f"{output_dir}/{ckpt_name}_F-{max_num_frames}_fps-{tgt_video_fps}_pool-{pool_size}.json"
    # pred_filepath = "output_debug/tmp.json"
    if model.config.slow_fast_path_flag:
        sf_model_name = f"SF-s{model_args.slow_mm_spatial_pool_stride}-t{model_args.slow_path_stride}"
        pred_filepath = pred_filepath.replace(".json", f"_{sf_model_name}.json")
    os.makedirs(os.path.dirname(pred_filepath), exist_ok=True)

    return model, tokenizer, data_root, pred_filepath


if __name__ == "__main__":    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()
    if eval_args.reproduce:
        set_cuda_deterministic()
    
    gen_kwargs = {
        "max_new_tokens": 32,
        "num_beams": 1,
        "do_sample": False,
        "use_cache": True,
        "num_logits_to_keep": 1,
    }
    
    ## setup data path and evaluator
    dataset_name = eval_args.dataset_name
    anno_filepath, video_root, dataset_dir = get_data_path(dataset_name)
    evaluate = get_evaluator(dataset_dir)

    ## setup model
    model, tokenizer, data_root, pred_filepath = setup_llava_video_model(model_args, data_args, eval_args)
    model.eval()

    ## create recording file pointers
    pred_filepath_tmp = pred_filepath.replace(".json", "_tmp.json")
    prev_pred_qids = None
    if os.path.exists(pred_filepath_tmp):
        pred_pred_df = pd.read_json(path_or_buf=pred_filepath_tmp, lines=True, dtype={'question_id': str})
        if len(pred_pred_df) > 0:
            prev_pred_qids = pred_pred_df['question_id'].values
            pred_file = open(pred_filepath_tmp, "a")
        else:
            pred_file = open(pred_filepath_tmp, "w")
    else:
        pred_file = open(pred_filepath_tmp, "w")

    runtime_pred_filepath_tmp = pred_filepath.replace(".json", "_runtime_tmp.json")
    if os.path.exists(runtime_pred_filepath_tmp):
        runtime_pred_file_tmp = open(runtime_pred_filepath_tmp, "a")
    else:
        runtime_pred_file_tmp = open(runtime_pred_filepath_tmp, "w")
    
    prompt_stat_filepath_tmp = pred_filepath.replace(".json", "_prompt_stat_tmp.json")
    if os.path.exists(prompt_stat_filepath_tmp):
        prompt_stat_file_tmp = open(prompt_stat_filepath_tmp, "a")
    else:
        prompt_stat_file_tmp = open(prompt_stat_filepath_tmp, "w")

    ## build dataset and dataloader        
    vidqa_feature_dataset = VidQA_Loader_Feature(dataset_name, anno_filepath, data_root, tokenizer, data_args, answer_flag=True, prev_pred_qids=prev_pred_qids, first_sample=False)
    vidqa_feature_dataloader = DataLoader(vidqa_feature_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=3)

    ## run MLLM and save results
    for idx, d in enumerate(tqdm(vidqa_feature_dataloader)):        
        input_ids = d['input_ids'][0].unsqueeze(0).to("cuda")
        images = [d['feature'][0].to(torch.bfloat16).to("cuda")]
        image_sizes = [None]
        modalities = [d['modality'][0]]
        prompt_stat = {"sys": d['sys_len'][0].item(), "inst": d['inst_len'][0].item(), "frame": d['frame_len'][0].item()} # assume a single batch size
        stop_str = tokenizer.conv_tmpl.sep if tokenizer.conv_tmpl.sep_style != SeparatorStyle.TWO else tokenizer.conv_tmpl.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        if idx == 0:
            ## warm-up gpu for robust latency measure
            with torch.inference_mode():
                _ = model.generate(
                    inputs=input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    modalities=modalities,
                    stopping_criteria=[stopping_criteria],
                    prompt_stat=prompt_stat,
                    **gen_kwargs
                )

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.inference_mode():
            output_ids, runtime_dict = model.generate(
                inputs=input_ids,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                stopping_criteria=[stopping_criteria],
                prompt_stat=prompt_stat,
                **gen_kwargs
            )
        torch.cuda.synchronize()
        runtime_dict['all'] = time.time() - start_time
        runtime_dict['qid'] = d['id'][0]
        
        ## prompt stat 
        prompt_stat['merged_video'] = prompt_stat['num_last_layer_token'] - prompt_stat['sys'] - prompt_stat['inst']
        prompt_stat['visual_merged_ratio'] = 100 * prompt_stat['merged_video'] / prompt_stat['video']
        prompt_stat['input_merged_ratio'] = 100 * prompt_stat['num_last_layer_token'] / (prompt_stat['video'] + prompt_stat['sys'] + prompt_stat['inst'])
        prompt_stat_output = { 
                                "qid": d['id'][0],
                                "visual_merged_ratio": prompt_stat["visual_merged_ratio"],
                                "input_merged_ratio": prompt_stat["input_merged_ratio"],
                                "visual_token_ori": prompt_stat["video"],
                                "visual_token_merged": prompt_stat["merged_video"]
                            }
    
        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        pred_file.write(json.dumps({"question_id": d['id'][0], "answer": text_outputs, "gt": d['answer'][0]}) + "\n")
        pred_file.flush()
        runtime_pred_file_tmp.write(json.dumps(runtime_dict) + "\n")
        runtime_pred_file_tmp.flush()
        prompt_stat_file_tmp.write(json.dumps(prompt_stat_output) + "\n")
        prompt_stat_file_tmp.flush()
    
    pred_file.close()
    runtime_pred_file_tmp.close()
    prompt_stat_file_tmp.close()

    runtime_list = pd.read_json(path_or_buf=runtime_pred_filepath_tmp, lines=True, dtype={'qid': str}).to_dict('records')
    runtime_pred_filepath = pred_filepath.replace(".json", "_runtime.pkl")
    with open(runtime_pred_filepath, "wb") as fp:
        pickle.dump(runtime_list, fp)

    prompt_stat_list = pd.read_json(path_or_buf=prompt_stat_filepath_tmp, lines=True, dtype={'qid': str}).to_dict('records')
    prompt_stat_filepath = pred_filepath.replace(".json", "_prompt_stat.pkl")
    with open(prompt_stat_filepath, "wb") as fp:
        pickle.dump(prompt_stat_list, fp)
    
    results = pd.read_json(path_or_buf=pred_filepath_tmp, lines=True, dtype={'question_id': str}).to_dict('records')
    with open(pred_filepath, 'w') as fp:
        json.dump(results, fp, indent=2)
    
    evaluate(pred_filepath, anno_filepath)
    measure_runtime(runtime_pred_filepath, anno_filepath, save_flag=True)
    measure_prompt_stat(prompt_stat_filepath, anno_filepath, save_flag=True)
