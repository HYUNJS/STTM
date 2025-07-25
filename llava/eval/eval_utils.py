import json
import os
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import torch

from llava.eval.metric_vnbench import evaluate as evaluate_vnbench
from llava.eval.metric_videomme import evaluate as evaluate_videomme
from llava.eval.metric_egoschema import evaluate as evaluate_egoschema
from llava.eval.metric_longvideobench import evaluate as evaluate_longvideobench
from llava.eval.metric_nextqa import evaluate as evaluate_nextqa
from llava.eval.metric_mlvu_mcq import evaluate as evaluate_mlvu_mcq

from llava.utils import rank0_print
from token_merging_utils.monkey_patch_interface import replace_qwen2_by_sparse_attn


@dataclass
class EvalArguments:
    dataset_name: str
    repeat_idx: Optional[int] = field(default=0)
    reproduce: Optional[bool] = field(default=False)

def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_data_path(dataset_name):
    if dataset_name == "videomme":
        anno_filepath = "datasets/videomme/annotations/videomme.json"
        video_root = "datasets/videomme/videos/"
        dataset_dir = "videomme"
    elif dataset_name == "vnbench":
        anno_filepath = "datasets/vnbench/annotations/VNBench-main-4try_v2.json"
        video_root = "datasets/vnbench/videos/"
        dataset_dir = "vnbench"
    elif dataset_name == "egoschema":
        anno_filepath = "datasets/egoschema/annotations/egoschema_fullset_v2.json"
        video_root = "datasets/egoschema/videos/"
        dataset_dir = "egoschema"
    elif dataset_name == "lvb-val":
        anno_filepath = "datasets/longvideobench/annotations/lvb_val_v2.json"
        video_root = "datasets/longvideobench/videos/"
        dataset_dir = "longvideobench"
    elif dataset_name == "lvb-test":
        anno_filepath = "datasets/longvideobench/annotations/lvb_test_v2.json"
        video_root = "datasets/longvideobench/videos/"
        dataset_dir = "longvideobench"
    elif dataset_name == "nextqa-mcq":
        anno_filepath = "datasets/nextqa/annotations/MC_test_v2.json"
        video_root = "datasets/nextqa/videos/"
        dataset_dir = "nextqa"
    elif dataset_name == "mlvu-mcq":
        anno_filepath = "datasets/mlvu/annotations/MLVU_mcq_v2.json"
        video_root = "datasets/mlvu/videos/"
        dataset_dir = "mlvu"
    else:
        raise NotImplementedError(f"{dataset_name} is not defined")
    return anno_filepath, video_root, dataset_dir

def get_evaluator(dataset_dir):
    mapper = {
        "videomme": evaluate_videomme,
        "vnbench": evaluate_vnbench,
        "egoschema": evaluate_egoschema,
        "longvideobench": evaluate_longvideobench,
        "nextqa": evaluate_nextqa,
        "mlvu": evaluate_mlvu_mcq,
    }
    return mapper[dataset_dir]

def get_prompt_stat(input_ids, target_id):
    ## input_ids: Tensor[N]
    ## Assume a single image or video input. Provide the prompt length based on the given index
    target_idxs = torch.where(input_ids == target_id)[0]
    target_start_idx = target_idxs[0].item()
    target_end_idx = target_idxs[-1].item()
    # target_pos = torch.where(input_ids == target_id)[0].item() # not supporting multiple target token case
    prompt_length = {
        "sys": target_start_idx,
        "inst": len(input_ids) - (target_end_idx + 1)
    }
    return prompt_length

def set_cuda_deterministic():
    # Tell cuDNN to only use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # And (if you’re on PyTorch ≥1.8) force all PyTorch ops to be deterministic
    torch.use_deterministic_algorithms(True)
    ## Note, native matmul operation (@) requires to set "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_sa_cfg(model_args):
    sa_pattern = model_args.sa_pattern
    # assert sa_pattern in ["", "asa", "vsa", "bsa", "framefusion", "fastv", "quadtree", "octree"]

    sa_flag = sa_pattern != ""
    sa_start_layer_idx = model_args.sa_start_layer_idx
    bsa_topk = model_args.sa_bsa_topk
    bsa_topkp = model_args.sa_bsa_topkp
    bsa_bs = model_args.sa_bsa_bs
    asa_n_init = model_args.sa_asa_n_init
    asa_n_local = model_args.sa_asa_n_local
    sa_asa_n_ratio = model_args.sa_asa_n_ratio
    sa_tree_root_level = model_args.sa_tree_root_level
    sa_tree_thresh = model_args.sa_tree_thresh
    sa_tree_temporal_thresh = model_args.sa_tree_temporal_thresh
    sa_tree_weighted_avg = model_args.sa_tree_weighted_avg
    sttm_slow_ver = model_args.sttm_slow_ver
    sim_per_head = model_args.sim_per_head
    
    sa_fastv_evict_ratio = model_args.sa_fastv_evict_ratio
    sa_framefusion_cost = model_args.sa_framefusion_cost
    
    sa_tree_dist_topk = model_args.sa_tree_dist_topk
    sa_tree_dist_time = model_args.sa_tree_dist_time
    sa_tree_trk_thresh = model_args.sa_tree_trk_thresh
    sa_tree_trk_layer_idx = model_args.sa_tree_trk_layer_idx

    pos_emb_ver = model_args.pos_emb_ver
    pos_emb_weighted_avg = model_args.pos_emb_weighted_avg

    ## for ToMe
    sa_prune_ratio = model_args.sa_prune_ratio
    sa_tome_ver = model_args.sa_tome_ver

    ## for DyCoke
    dycoke_l = model_args.dycoke_l
    dycoke_p = model_args.dycoke_p

    if sa_pattern in ["asa", "vsa", "bsa"]:
        if sa_pattern == "asa":
            sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "n_init": asa_n_init, "n_local": asa_n_local}
            output_dir = f"outputs_asa_layer-{sa_start_layer_idx}_n-init-{asa_n_init}_n-local-{asa_n_local}"
        elif sa_pattern == "bsa":
            sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "bsa_topk": bsa_topk, "bsa_topkp": bsa_topkp, "bsa_bs_M": bsa_bs, "bsa_bs_N": bsa_bs}
            output_dir = f"outputs_bsa_bs-{bsa_bs}_layer-{sa_start_layer_idx}_topk-{bsa_topk}"
            if bsa_topkp > 0.0:
                output_dir = f"{output_dir}_topkp-{bsa_topkp:.3f}"
        elif sa_pattern == "vsa":
            pass
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif "quadtree" in sa_pattern:
        sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "sa_tree_thresh": sa_tree_thresh, "sa_tree_root_level": sa_tree_root_level, "sa_tree_temporal_thresh": sa_tree_temporal_thresh,
                    "sa_tree_weighted_avg": sa_tree_weighted_avg, "sa_tree_dist_topk": sa_tree_dist_topk, "sa_tree_dist_time": sa_tree_dist_time, "sa_tree_trk_thresh": sa_tree_trk_thresh, "sa_tree_trk_layer_idx": sa_tree_trk_layer_idx,
                    "sttm_slow_ver": sttm_slow_ver, "sim_per_head": sim_per_head, "pos_emb_ver": pos_emb_ver, "pos_emb_weighted_avg": pos_emb_weighted_avg}
        output_dir = f"outputs_{sa_pattern}_layer-{sa_start_layer_idx}_thresh-{sa_tree_thresh:.3f}_root-level-{sa_tree_root_level}"
        if sa_tree_temporal_thresh > 0:
            output_dir = f"{output_dir}_tempo-thresh-{sa_tree_temporal_thresh:.3f}"
        if sa_tree_weighted_avg:
            output_dir = f"{output_dir}_weighted-avg"
        ##### Tracklet merging case
        if sa_tree_trk_layer_idx > 0:
            if sa_tree_dist_topk > 0:
                output_dir = f"{output_dir}_st-merge_layer-{sa_tree_trk_layer_idx}_dist-topk-{sa_tree_dist_topk}_trk-thresh-{sa_tree_trk_thresh:.3f}"
            elif sa_tree_dist_time > 0:
                output_dir = f"{output_dir}_st-merge_layer-{sa_tree_trk_layer_idx}_dist-time-{sa_tree_dist_time}_trk-thresh-{sa_tree_trk_thresh:.3f}"
        ##### 
        if sttm_slow_ver:
            output_dir += "_slow"
        if sim_per_head:
            output_dir += "_sim-per-head"
        
        if "quadtree-abl-" in sa_pattern:
            pos_emb_weighted_avg_flag = 1 if pos_emb_weighted_avg else 0
            output_dir += f"_pos-ver-{pos_emb_ver}-weighted-{pos_emb_weighted_avg_flag}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "octree":
        sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "sa_tree_thresh": sa_tree_thresh, "sa_tree_root_level": sa_tree_root_level}
        output_dir = f"outputs_octree_layer-{sa_start_layer_idx}_thresh-{sa_tree_thresh:.3f}_root-level-{sa_tree_root_level}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "framefusion":
        output_dir = f"outputs_framefusion_cost-{sa_framefusion_cost:.3f}" # apply monkey patch in main
    elif sa_pattern == "fastv":
        output_dir = f"outputs_fastv_layer-{sa_start_layer_idx}_ratio-{sa_fastv_evict_ratio:.3f}" # apply monkey patch in main
    elif sa_pattern == "framefusion-merge":
        output_dir = f"outputs_framefusion-merge_layer-{sa_start_layer_idx}_ratio-{sa_prune_ratio:.3f}" # apply monkey patch in main
    elif sa_pattern == "streamingllm":
        output_dir = f"outputs_streamingllm_ninit-{asa_n_init}_ratio-{sa_asa_n_ratio:.3f}" # apply monkey patch in main
    elif sa_pattern == "pyrd":
        sa_pyrd_loc_list_str = model_args.sa_pyrd_loc_list.replace(" ", "")
        sa_pyrd_size_list_str = model_args.sa_pyrd_size_list.replace(" ", "")
        sa_pyrd_loc_list = [int(v) for v in sa_pyrd_loc_list_str.split(",")]
        sa_pyrd_size_list = [int(v) for v in sa_pyrd_size_list_str.split(",")]
        sa_kwargs = {"sa_pyrd_loc_list": sa_pyrd_loc_list, "sa_pyrd_size_list": sa_pyrd_size_list}
        output_dir = f"outputs_pyrd_layer-{sa_pyrd_loc_list_str}_size-{sa_pyrd_size_list_str}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "tome":
        sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "sa_prune_ratio": sa_prune_ratio, "sa_tome_ver": sa_tome_ver}
        output_dir = f"outputs_tome_layer-{sa_start_layer_idx}_ratio-{sa_prune_ratio:.3f}_ver-{sa_tome_ver}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "dycoke-stage1":
        sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "sa_prune_ratio": sa_prune_ratio}
        output_dir = f"outputs_dycoke-stage1_layer-{sa_start_layer_idx}_ratio-{sa_prune_ratio:.3f}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "dycoke":
        sa_kwargs = {"sa_start_layer_idx": sa_start_layer_idx, "sa_prune_ratio": sa_prune_ratio, "dycoke_l": dycoke_l, "dycoke_p": dycoke_p}
        output_dir = f"outputs_dycoke_layer-{sa_start_layer_idx}_ratio-{sa_prune_ratio:.3f}_l-{dycoke_l}_p-{dycoke_p:.3f}"
        replace_qwen2_by_sparse_attn(sa_pattern, **sa_kwargs)
    elif sa_pattern == "":
        output_dir = f"outputs_original"
    else:
        raise NotImplementedError(f"{sa_pattern} attention is not yet defined")

    return output_dir

def format_videomme(data_list, answer_flag=False):
    rank0_print("Format VideoMME annotations for llava-vid-Qwen2")
    new_data_list = []

    for anno in data_list:
        vid = anno['videoID']
        video_filepath = f"{vid}.mp4"
        qid = anno['question_id']
        question = anno['question']
        options = anno['options']
        answer = anno['answer'] if answer_flag else None

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        option = "\n".join([f"{opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list

def format_vnbench(data_list, answer_flag=False):
    rank0_print("Format VNBench annotations for llava-vid-Qwen2")
    new_data_list = []

    for anno in data_list:
        vid = anno['videoID']
        video_filepath = f"{vid}.mp4"
        qid = anno['question_id']
        question = anno['question']
        options = anno['options']
        try_id = anno['try']
        needle_time = anno['needle_time']
        answer = anno['answer'] if answer_flag else None

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        option = "\n".join([f"{opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "try": try_id,
            "needle_time": needle_time,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list

def format_egoschema(data_list, answer_flag=False):
    rank0_print("Format EgoSchema annotations for llava-vid-Qwen2")
    new_data_list = []
    has_answer = "answer" in data_list[0]
    for anno in data_list:
        vid = anno['q_uid']
        video_filepath = f"{vid}.mp4"
        qid = anno['q_uid']
        question = anno['question']
        options = [anno['option 0'], anno['option 1'], anno['option 2'], anno['option 3'], anno['option 4']]
        options_prefix = ["A", "B", "C", "D", "E"]
        answer = anno['answer'] if answer_flag and has_answer else ""

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, or E) of the correct option."
        option = "\n".join([f"{options_prefix[i]}. {opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list

def format_lvb(data_list, answer_flag=False):
    rank0_print("Format LongVideoBench annotations for llava-vid-Qwen2")
    new_data_list = []

    for anno in data_list:
        vid = anno['videoID']
        video_filepath = f"{vid}.mp4"
        qid = anno['question_id']
        question = anno['question']
        options = anno['options']
        options_prefix = ["A", "B", "C", "D", "E", "F"] # max 6 options
        answer = anno['answer'] if answer_flag else None

        option_choices = ["", "", "(A, B, or C)", "(A, B, C, or D)", "(A, B, C, D, or E)", "(A, B, C, D, E, or F)"]
        option_prompt = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter {option_choices[len(options)]} of the correct option."
        option = "\n".join([f"{options_prefix[i]}. {opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list

def format_nextqa_mcq(data_list, answer_flag=False):
    rank0_print("Format nextqa mcq annotations for llava-vid-Qwen2")
    new_data_list = []
    has_answer = "answer" in data_list[0]
    for anno in data_list:
        vid = anno['video_id']
        video_filepath = f"{vid}.mp4"
        qid = anno['question_id']
        question = anno['question']
        options = anno['options']
        options_prefix = ["A", "B", "C", "D", "E"]
        answer = anno['answer'] if answer_flag and has_answer else ""

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, D, or E) of the correct option."
        option = "\n".join([f"{options_prefix[i]}. {opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list


def format_mlvu_mcq(data_list, answer_flag=False):
    rank0_print("Format MLVU annotations for llava-vid-Qwen2")
    new_data_list = []
    has_answer = "answer" in data_list[0]
    for anno in data_list:
        vid = anno['video_id']
        video_filepath = f"{vid}.mp4"
        qid = anno['question_id']
        question = anno['question']
        options = anno['options']
        options_prefix = ["A", "B", "C", "D"]
        answer = anno['answer'] if answer_flag and has_answer else ""

        option_prompt = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
        option = "\n".join([f"{options_prefix[i]}. {opt}" for i, opt in enumerate(options)])
        post_prompt = "The best answer is:"
        full_prompt = option_prompt + "\n" + question + "\n" + option + "\n" + post_prompt
        conversations = full_prompt

        data = {
            "qid": qid,
            "vid": vid,
            "video_filepath": video_filepath,
            "conversations": conversations,
            "answer": answer,
            "question": question,
            "options": options,
            }
        new_data_list.append(data)

    return new_data_list
