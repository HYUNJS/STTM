import copy
import torch
import os
from tqdm import tqdm
import math
import transformers
import pickle
import sys
import einops
from torch.utils.data import DataLoader
## Get the absolute path of the project directory
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
## Add the project directory to sys.path
sys.path.append(project_dir)

# from llava.train.train import ModelArguments, DataArguments
from llava.utils import ModelArguments, DataArguments
from llava.eval.eval_utils import EvalArguments, get_data_path
from llava.eval.video_dataset import Video_Loader
# from llava.model.qwen2vl.modeling_qwen2vl import Qwen2VLForConditionalGeneration
# from transformers import Qwen2VLProcessor
from transformers import Gemma3Processor, Gemma3ForConditionalGeneration


import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.nn\.modules\.module"
)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    ## load pretrained model and preprocessor
    dtype = torch.bfloat16
    model_base = model_args.model_base
    model_path = model_args.model_name_or_path
    ckpt_name = model_path.split("/")[1].lower()
    model = Gemma3ForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, attn_implementation="flash_attention_2", device_map="auto")
    processor = Gemma3Processor.from_pretrained(model_path)
    data_args.processor = processor
    data_args.image_processor = processor.image_processor
    model.eval()
    vision_encoder = model.model.vision_tower
    multi_modal_projector = model.model.multi_modal_projector
    del model
    torch.cuda.empty_cache()
    
    ## get data cfg
    max_num_frames = data_args.frames_upbound
    tgt_video_fps = data_args.tgt_video_fps
    data_args.add_time_instruction = False
    dataset_name = eval_args.dataset_name
    anno_filepath, video_root, dataset_dir = get_data_path(dataset_name)

    data_root = ckpt_name if model_base is None else model_base.split("/")[1].lower()
    data_root = os.path.join(video_root.replace("/videos", ""), "preprocess_data", data_root, f"F-{max_num_frames}_fps-{tgt_video_fps}")
    feature_dir = os.path.join(data_root, "features")
    metadata_dir = os.path.join(data_root, "metadata")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    prev_vids = [d.replace(".pt", "") for d in os.listdir(feature_dir)]

    ## build dataloader
    video_dataset = Video_Loader(dataset_name, anno_filepath, video_root, data_args, prev_vids)
    video_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=12, prefetch_factor=4)
    
    ## run MLLM and save extracted features
    with torch.inference_mode():
        for idx, (frames, metadata) in enumerate(tqdm(video_dataloader)):
            frames = frames[0].to(dtype).to("cuda")
            num_frames = len(frames)
            vision_outputs = vision_encoder(pixel_values=frames).last_hidden_state
            video_features = multi_modal_projector(vision_outputs) # [T, HW, C]
            video_features = video_features.cpu()

            ## save
            vid = metadata['vid'][0]
            feat_filepath = os.path.join(feature_dir, f"{vid}.pt")
            meta_filepath = os.path.join(metadata_dir, f"{vid}.pkl")
            torch.save(video_features, feat_filepath)
            metadata['vid'] = vid
            metadata['video_time'] = metadata['video_time'].item()
            metadata['frame_time'] = metadata['frame_time'][0]
            metadata['num_frames'] = metadata['num_frames'].item()
            with open(meta_filepath, "wb") as fp:
                pickle.dump(metadata, fp)
