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
import einops

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
from llava.model.framefusion.models.qwen2vl.modeling_qwen2vl_streamingllm import replace_Qwen2VL_streamingllm
from llava.model.framefusion.models.qwen2vl.modeling_qwen2vl_fastv import replace_Qwen2VL_fastv
from llava.model.framefusion.models.qwen2vl.modeling_qwen2vl_merging import replace_Qwen2VL_merging

from llava.train.train import ModelArguments, DataArguments
from llava.eval.eval_utils import set_cuda_deterministic, parse_sa_cfg, get_data_path, get_evaluator, EvalArguments
from llava.eval.video_dataset import VidQA_Loader_Feature_Qwen2VL
from llava.model.qwen2vl.modeling_qwen2vl import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLProcessor


## set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    
    model_base = model_args.model_base
    model_path = model_args.model_name_or_path
    ckpt_name = model_path.split("/")[1].lower()
    max_num_frames = data_args.frames_upbound
    tgt_video_fps = data_args.tgt_video_fps
    repeat_idx = eval_args.repeat_idx
    
    dataset_name = eval_args.dataset_name
    anno_filepath, video_root, dataset_dir = get_data_path(dataset_name)
    evaluate = get_evaluator(dataset_dir)
    data_root = ckpt_name if model_base is None else model_base.split("/")[1].lower()
    data_root = os.path.join(f"datasets/{dataset_dir}/preprocess_data", data_root, f"F-{max_num_frames}_fps-{tgt_video_fps}")
    
    ## set output_dir
    output_dir = parse_sa_cfg(model_args)
    if repeat_idx > 0:
        output_dir = f"outputs_{dataset_name}_repeat-{repeat_idx}/{output_dir}"
    else:
        output_dir = f"outputs_{dataset_name}/{output_dir}"
    # output_dir = "outputs_debug"

    if eval_args.reproduce:
        output_dir = os.path.join("outputs_reproduce", output_dir)
    else:
        output_dir = os.path.join("outputs", output_dir)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    processor = Qwen2VLProcessor.from_pretrained(model_path)
    data_args.processor = processor
    
    if model_args.sa_pattern == "framefusion":
        # llava-based: cost = 0.3, similarity_lower_bound = 0.6, ratio_lower_bound = 0.1
        model = apply_framefusion(model, cost=model_args.sa_framefusion_cost, similarity_lower_bound=0.6, ratio_lower_bound=0.1)
    elif model_args.sa_pattern == "fastv":
        replace_Qwen2VL_fastv(model, fastv_k=model_args.sa_start_layer_idx, fastv_r=model_args.sa_fastv_evict_ratio)
    elif model_args.sa_pattern == "framefusion-merge":
        sparsity = [0.0] * 28
        sparsity[model_args.sa_start_layer_idx] = model_args.sa_prune_ratio
        replace_Qwen2VL_merging(model, sparsity=sparsity)
    elif model_args.sa_pattern == "streamingllm":
        replace_Qwen2VL_streamingllm(model, init_num=model_args.sa_asa_n_init, length_rate=model_args.sa_asa_n_ratio)
    


    # Create answer file
    pred_filepath = f"{output_dir}/{ckpt_name}_F-{max_num_frames}_fps-{tgt_video_fps}.json"
    # pred_filepath = "output/tmp.json"
    os.makedirs(os.path.dirname(pred_filepath), exist_ok=True)
    
    pred_filepath_tmp = pred_filepath.replace(".json", "_tmp.json")
    prev_pred_qids = None
    if os.path.exists(pred_filepath_tmp):
        pred_pred_df = pd.read_json(path_or_buf=pred_filepath_tmp, lines=True)
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
    vidqa_feature_dataset = VidQA_Loader_Feature_Qwen2VL(dataset_name, anno_filepath, data_root, data_args, answer_flag=True, prev_pred_qids=prev_pred_qids, first_sample=False)
    vidqa_feature_dataloader = DataLoader(vidqa_feature_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=3)
    model.eval()

    ## run MLLM and save results
    for idx, d in enumerate(tqdm(vidqa_feature_dataloader)):        
        input_ids = d['input_ids'][0].unsqueeze(0).to("cuda")
        video_features = d['feature'][0].to(torch.bfloat16).to("cuda")
        attention_mask = d['attention_mask'][0].unsqueeze(0).to("cuda")
        video_grid_thw = d['video_grid_thw'][0].to("cuda")
        model_inputs = {"input_ids": input_ids, "video_features": video_features, "attention_mask": attention_mask, "video_grid_thw": video_grid_thw}
        prompt_stat = {"sys": d['sys_len'][0].item(), "inst": d['inst_len'][0].item(), "frame": d['frame_len'][0].item()} # assume a single batch size
        T, H, W, _ = video_features.shape
        prompt_stat.update({"video": T*H*W, "T": T, "H": H, "W": W})
        
        with torch.inference_mode():
            inputs_embeds = model.model.embed_tokens(input_ids)
            video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds) # [B, N, C]
            video_embeds = einops.rearrange(video_features, "t h w c -> (t h w) c") # [THW, C]
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if idx == 0:
            ## warm-up gpu for robust latency measure
            with torch.inference_mode():
                _ = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                video_grid_thw=video_grid_thw, prompt_stat=prompt_stat, **gen_kwargs)

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.inference_mode():
            generated_ids, runtime_dict = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                           video_grid_thw=video_grid_thw, prompt_stat=prompt_stat, **gen_kwargs)
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
    
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        text_outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_file.write(json.dumps({"question_id": d['id'][0], "answer": text_outputs[0], "gt": d['answer'][0]}) + "\n")
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
