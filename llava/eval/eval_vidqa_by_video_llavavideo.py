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
from torch.utils.data import Dataset, Sampler, DataLoader
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

from llava.conversation import SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria
from llava.train.train import ModelArguments, DataArguments, TrainingArguments
from llava.eval.eval_utils import set_cuda_deterministic, parse_sa_cfg, get_data_path, get_evaluator, EvalArguments
from llava.eval.video_dataset import VidQA_Loader_Video
from llava.eval.eval_vidqa_by_feat_llavavideo import setup_llava_video_model


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
    vidqa_loader = VidQA_Loader_Video(dataset_name, anno_filepath, video_root, tokenizer, data_args, answer_flag=True, prev_pred_qids=prev_pred_qids)
    
    ## run MLLM and save results
    for idx, d in enumerate(tqdm(vidqa_loader)):
        input_ids = d['input_ids'].unsqueeze(0).to("cuda")
        images = [d['image'].to(torch.bfloat16).to("cuda")]
        image_sizes = [None]
        modalities = [d['modality']]
        prompt_stat = {"sys": d['sys_len'], "inst": d['inst_len'], "frame": d['frame_len']} # assume a single batch size
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
        runtime_dict['qid'] = d['id']
        
        ## prompt stat 
        prompt_stat['merged_video'] = prompt_stat['num_last_layer_token'] - prompt_stat['sys'] - prompt_stat['inst']
        prompt_stat['visual_merged_ratio'] = 100 * prompt_stat['merged_video'] / prompt_stat['video']
        prompt_stat['input_merged_ratio'] = 100 * prompt_stat['num_last_layer_token'] / (prompt_stat['video'] + prompt_stat['sys'] + prompt_stat['inst'])
        prompt_stat_output = { 
                                "qid": d['id'],
                                "visual_merged_ratio": prompt_stat["visual_merged_ratio"],
                                "input_merged_ratio": prompt_stat["input_merged_ratio"],
                                "visual_token_ori": prompt_stat["video"],
                                "visual_token_merged": prompt_stat["merged_video"]
                            }

        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        pred_file.write(json.dumps({"question_id": d['id'], "answer": text_outputs, "gt": d['answer']}) + "\n")
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
