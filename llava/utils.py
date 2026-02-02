import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import math
import requests
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field

from llava.constants import LOGDIR

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_resampler_model_name: Optional[str] = field(default=None)
    mm_resampler_skip_layer_flag: Optional[bool] = field(default=False)
    mm_resampler_post_flag: Optional[bool] = field(default=False)
    mm_resampler_grid_width: Optional[int] = field(default=0) # "-1: using all input tokens; X: indexing tokens per X interval - e.g., 15")
    mm_resampler_skip_last_norm: Optional[bool] = field(default=False)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    max_batch_size: Optional[int] = field(default=256)
    model_base: Optional[str] = field(default=None)

    slow_mm_spatial_pool_stride: Optional[int] = field(default=2)
    slow_path_stride: Optional[int] = field(default=10)
    slow_fast_path_flag: Optional[bool] = field(default=False)

    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    ## Sparse attention common config
    sa_pattern: Optional[str] = field(default="")
    sa_start_layer_idx: Optional[int] = field(default=0) # 0 1 ...
    ## Block sparse attention
    sa_bsa_topk: Optional[int] = field(default=100) # 10 50 100
    sa_bsa_topkp: Optional[float] = field(default=0.0) #
    sa_bsa_bs: Optional[int] = field(default=64) # 64
    ## A-shape attention
    sa_asa_n_init: Optional[int] = field(default=8) # 64
    sa_asa_n_local: Optional[int] = field(default=3968) # 64
    sa_asa_n_ratio: Optional[float] = field(default=0.5)
    ## Tree token merging attention
    sa_tree_root_level: Optional[int] = field(default=0)
    sa_tree_thresh: Optional[float] = field(default=-1.0)
    sa_tree_temporal_thresh: Optional[float] = field(default=-1.0)
    sa_tree_weighted_avg: Optional[bool] = field(default=False)
    sa_tree_dist_topk: Optional[int] = field(default=-1)
    sa_tree_dist_time: Optional[int] = field(default=-1)
    sa_tree_trk_thresh: Optional[float] = field(default=-1.0)
    sa_tree_trk_layer_idx: Optional[int] = field(default=-1)
    sttm_slow_ver: Optional[bool] = field(default=False)
    sim_per_head: Optional[bool] = field(default=False)
    pos_emb_ver: Optional[int] = field(default=0)
    pos_emb_weighted_avg: Optional[bool] = field(default=False)
    ## FastV
    sa_fastv_evict_ratio: Optional[float] = field(default=0.50)
    ## FrameFusion
    sa_framefusion_cost: Optional[float] = field(default=0.30)
    ## pyrd merging - currently, only assume a single merging, instead of iterative merging
    sa_pyrd_loc_list: Optional[str] = field(default="2")
    sa_pyrd_size_list: Optional[str] = field(default="10")
    ## ToMe
    sa_prune_ratio: Optional[float] = field(default=0.50)
    sa_tome_ver: Optional[str] = field(default="frame")
    ## Dycoke
    dycoke_l: Optional[int] = field(default=3)
    dycoke_p: Optional[float] = field(default=0.8)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    tgt_video_fps: Optional[int] = field(default=None)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)


def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    if hasattr(data_args, "tgt_video_fps") and data_args.tgt_video_fps:
        tgt_fps = data_args.tgt_video_fps
        vid_fps = vr.get_avg_fps()
        intv = vid_fps / tgt_fps
        frame_idx = np.ceil(np.arange(math.ceil((total_frame_num-1) / intv)) * intv)
        if total_frame_num == 1 or frame_idx[-1] != (total_frame_num - 1):
            frame_idx = np.concatenate([frame_idx, np.array([total_frame_num-1])])

        if data_args.frames_upbound > 0 and len(frame_idx) > data_args.frames_upbound:
            ## if max_frames is set and tgt_fps sampling exceeds max_frames, bound it my max_frames
            frame_idx = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
        frame_time = frame_idx / vid_fps

    else:
        avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
        frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
        frame_time = [i/avg_fps for i in frame_idx]
        if data_args.frames_upbound > 0:
            if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
