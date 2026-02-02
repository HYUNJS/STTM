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

from llava.eval.eval_vidqa_by_feat_qwen2vl import setup_qwen2vl_model

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
    model, processor, data_root, pred_filepath = setup_qwen2vl_model(model_args, data_args, eval_args, vis_flag=False)
    model.eval()
    vis_dir = pred_filepath.replace(".json", "")
    os.makedirs(vis_dir, exist_ok=True)
    
    ## build dataset and dataloader
    prev_pred_qids = None
    vidqa_feature_dataset = VidQA_Loader_Feature_Qwen2VL(dataset_name, anno_filepath, data_root, data_args, answer_flag=True, prev_pred_qids=prev_pred_qids, first_sample=False)
    vidqa_feature_dataloader = DataLoader(vidqa_feature_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=3)
    
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
        
        with torch.inference_mode():
            # node_metadata = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            #                                video_grid_thw=video_grid_thw, prompt_stat=prompt_stat, **gen_kwargs)
            node_metadata = model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                video_grid_thw=video_grid_thw, prompt_stat=prompt_stat, vis_flag=True, **gen_kwargs)
        
        qid = d['id'][0]
        with open(os.path.join(vis_dir, f"{qid}.pkl"), "wb") as fp:
            pickle.dump(node_metadata, fp)

        if idx == 10:
            break