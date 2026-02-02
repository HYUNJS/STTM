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

from llava.conversation import SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria
from llava.train.train import ModelArguments, DataArguments, TrainingArguments
from llava.eval.eval_utils import set_cuda_deterministic, parse_sa_cfg, get_data_path, get_evaluator, EvalArguments
from llava.eval.video_dataset import VidQA_Loader_Feature
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
    model, tokenizer, data_root, pred_filepath = setup_llava_video_model(model_args, data_args, eval_args, vis_flag=True)
    model.eval()
    vis_dir = pred_filepath.replace(".json", "")
    os.makedirs(vis_dir, exist_ok=True)

    ## build dataset and dataloader        
    prev_pred_qids = None
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

        with torch.inference_mode():
            node_metadata = model.generate(
                inputs=input_ids,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                stopping_criteria=[stopping_criteria],
                prompt_stat=prompt_stat,
                vis_flag=True,
                **gen_kwargs
            )

        qid = d['id'][0]
        with open(os.path.join(vis_dir, f"{qid}.pkl"), "wb") as fp:
            pickle.dump(node_metadata, fp)

        # if idx == 10:
        #     break