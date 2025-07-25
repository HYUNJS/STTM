import copy
import torch
import os
from tqdm import tqdm
from PIL import Image
import math
import transformers
import pickle
import sys
from torch.utils.data import DataLoader
## Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
## Add the project directory to sys.path
sys.path.append(project_dir)

from llava.train.train import ModelArguments, DataArguments
from llava.eval.eval_utils import EvalArguments, get_data_path
from llava.eval.video_dataset import Video_Loader
from llava.model.builder import load_pretrained_model
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, siglip_flash_forward

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.nn\.modules\.module"
)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    ## modify SigLip Attn to FlashAttn
    SigLipAttention.forward = siglip_flash_forward

    ## load pretrained model and preprocessor
    model_base = model_args.model_base
    model_path = model_args.model_name_or_path
    ckpt_name = model_path.split("/")[1].lower()
    model_name = "llava_qwen"
    conv_name = "qwen_2"
    overwrite_config = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, torch_dtype="bfloat16", overwrite_config=overwrite_config, device_map="auto")
    model.eval()
    model_cfg = copy.deepcopy(model.config)
    vision_encoder = model.get_vision_tower()
    del model
    torch.cuda.empty_cache()

    ## get data cfg
    max_num_frames = data_args.frames_upbound
    tgt_video_fps = data_args.tgt_video_fps
    dataset_name = eval_args.dataset_name
    anno_filepath, video_root, dataset_dir = get_data_path(dataset_name)

    data_root = ckpt_name if model_base is None else model_base.split("/")[1].lower()
    data_root = os.path.join(video_root.replace("/videos", ""), "preprocess_data", data_root, f"F-{max_num_frames}_fps-{tgt_video_fps}")
    feature_dir = os.path.join(data_root, "features")
    metadata_dir = os.path.join(data_root, "metadata")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    data_args.image_processor = image_processor
    data_args.add_time_instruction = model_cfg.add_time_instruction if hasattr(model_cfg, "add_time_instruction") else False
    data_args.is_multimodal = True
    data_args.mm_use_im_start_end = False
    
    max_batch_size = 512 ## max num_frame per vis_enc forward
    prev_vids = [d.replace(".pt", "") for d in os.listdir(feature_dir)]

    ## build dataloader
    video_dataset = Video_Loader(dataset_name, anno_filepath, video_root, data_args, prev_vids)
    video_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=12, prefetch_factor=4)

    ## run MLLM and save extracted features
    with torch.no_grad():
        for idx, (frames, metadata) in enumerate(tqdm(video_dataloader)):
            frames = frames[0].to(torch.bfloat16).to("cuda")
            num_frames = len(frames)
            num_batch = math.ceil(num_frames / max_batch_size)
            video_feats = []
            for b_i in range(num_batch):
                start_i = b_i*max_batch_size
                end_i = (b_i+1)*max_batch_size
                tgt_frames = frames[start_i:end_i]
                _video_feats = vision_encoder(tgt_frames)
                video_feats.append(_video_feats)
            video_feats = torch.cat(video_feats, dim=0)
            
            ## save
            vid = metadata['vid'][0]
            feat_filepath = os.path.join(feature_dir, f"{vid}.pt")
            meta_filepath = os.path.join(metadata_dir, f"{vid}.pkl")
            torch.save(video_feats.cpu(), feat_filepath)
            metadata['vid'] = vid
            metadata['video_time'] = metadata['video_time'].item()
            metadata['frame_time'] = metadata['frame_time'][0]
            metadata['num_frames'] = metadata['num_frames'].item()
            with open(meta_filepath, "wb") as fp:
                pickle.dump(metadata, fp)
