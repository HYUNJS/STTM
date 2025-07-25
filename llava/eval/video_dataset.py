import gc
import os
import sys
import copy
import pickle
import pandas as pd

import torch
import transformers
from torch.utils.data import Dataset
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

from llava.utils import rank0_print, process_video_with_decord
from llava.mm_utils import tokenizer_image_token
from llava.train.train import DataArguments
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.eval.eval_utils import (read_json, get_prompt_stat,
                                format_videomme, format_vnbench, format_lvb, format_egoschema, format_nextqa_mcq, format_mlvu_mcq)
from llava.model.qwen2vl.qwen_vl_utils import fetch_video


class Video_Loader(Dataset):
    def __init__(self, dataset_name, anno_filepath, video_root, data_args, prev_vids=None):
        
        self.dataset_name = dataset_name
        self.video_root = video_root
        self.data_args = data_args
        data_list = read_json(anno_filepath) # suppose anno_filepath in jsonl format
        vids = self.get_vids(dataset_name, data_list)
        if prev_vids is not None and len(prev_vids) > 0:
            vids = pd.Series(vids)
            vids = vids[~vids.isin(pd.Series(prev_vids))].values
        self.vids = vids

        rank0_print(f"Loading {anno_filepath}... {len(self.vids)} videos")
        if len(self.vids) == 0:
            sys.exit(0)
        
    def get_vids(self, dataset_name, data_list):
        if dataset_name == "videomme":
            vid_key = "videoID"
        elif dataset_name == "vnbench":
            vid_key = "videoID"
        elif dataset_name == "egoschema":
            vid_key = "q_uid"
        elif dataset_name == "lvb-val" or dataset_name == "lvb-test":
            vid_key = "videoID"
        elif dataset_name == "nextqa-mcq":
            vid_key = "video_id"
        elif dataset_name == "mlvu-mcq":
            vid_key = "video_id"
        else:
            raise NotImplementedError(f"{dataset_name} Formating for VideoLoader is not yet implemented")
        vids = pd.DataFrame(data_list)[vid_key].unique()
        return vids

    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, index):
        vid = self.vids[index]
        video_filepath = os.path.join(self.video_root, f"{vid}.mp4")
        frames, video_time, frame_time, num_frames = self.get_frames(video_filepath)
        video_metadata = {
            "vid": vid,
            "video_time": float(video_time),
            "frame_time": frame_time,
            "num_frames": num_frames,
        }
        return frames, video_metadata

    def get_frames(self, video_file):
        video_frames, video_time, frame_time, num_frames = process_video_with_decord(video_file, self.data_args)
        video_frames = self.data_args.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]            
        return video_frames, video_time, frame_time, num_frames


class Video_Loader_Qwen2VL(Dataset):
    def __init__(self, dataset_name, anno_filepath, video_root, data_args, prev_vids=None):
        
        self.dataset_name = dataset_name
        self.video_root = video_root
        self.data_args = data_args
        data_list = read_json(anno_filepath) # suppose anno_filepath in jsonl format
        vids = self.get_vids(dataset_name, data_list)
        if prev_vids is not None and len(prev_vids) > 0:
            vids = pd.Series(vids)
            vids = vids[~vids.isin(pd.Series(prev_vids))].values
        self.vids = vids

        rank0_print(f"Loading {anno_filepath}... {len(self.vids)} videos")
        if len(self.vids) == 0:
            sys.exit(0)
        
    def get_vids(self, dataset_name, data_list):
        if dataset_name == "videomme":
            vid_key = "videoID"
        elif dataset_name == "vnbench":
            vid_key = "videoID"
        elif dataset_name == "egoschema":
            vid_key = "q_uid"
        elif dataset_name == "lvb-val" or dataset_name == "lvb-test":
            vid_key = "videoID"
        elif dataset_name == "nextqa-mcq":
            vid_key = "video_id"
        elif dataset_name == "mlvu-mcq":
            vid_key = "video_id"
        else:
            raise NotImplementedError(f"{dataset_name} Formating for VideoLoader is not yet implemented")
        vids = pd.DataFrame(data_list)[vid_key].unique()
        return vids

    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, index):
        vid = self.vids[index]
        video_filepath = os.path.join(self.video_root, f"{vid}.mp4")
        frames, video_grid_thw, video_sample_fps, frame_time = self.get_frames(video_filepath)
        video_metadata = {
            "vid": vid,
            "video_sample_fps": video_sample_fps,
            "frame_time": frame_time,
            "video_grid_thw": video_grid_thw,
        }
        return frames, video_metadata

    def get_frames(self, video_file):
        vision_info = {"type": "video", "video": video_file, "fps": self.data_args.tgt_video_fps, "max_frames": self.data_args.frames_upbound}
        video_frames, video_sample_fps, frame_time = fetch_video(vision_info, return_video_sample_fps=True)
        videos_inputs = self.data_args.processor.image_processor(images=None, videos=video_frames, return_tensors="pt")
        video_frames = videos_inputs['pixel_values_videos']
        video_grid_thw = videos_inputs['video_grid_thw']
        return video_frames, video_grid_thw, video_sample_fps, frame_time


class VidQA_Loader_Video():
    def __init__(self, dataset_name: str, anno_filepath: str, video_root: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments,
                answer_flag: bool = False, prev_pred_qids=None):

        self.dataset_name = dataset_name
        self.video_root = video_root
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.answer_flag = answer_flag
        self._data_list = read_json(anno_filepath) # suppose anno_filepath in jsonl format
        if prev_pred_qids is not None:
            new_data_list = []
            for d in self._data_list:
                if d['question_id'] not in prev_pred_qids:
                    new_data_list.append(d)
            self._data_list = new_data_list

        rank0_print(f"Loading {anno_filepath}... {len(self._data_list)} samples ")
        if len(self._data_list) == 0:
            sys.exit(0)
        self.data_list = self.format_data(self._data_list)

        ## init frames memory (caching)
        ## remove it from memory if it is not retrieved for n times
        self.memory_thresh = 8
        self.memory_frames = {}
        self.memory_video_time = {}
        self.memory_frame_time = {}
        self.memory_num_frames = {}
        self.memory_counter = {}

    def format_data(self, data_list):
        if self.dataset_name == "videomme":
            return format_videomme(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench_short":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "egoschema":
            return format_egoschema(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-val":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-test":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "nextqa-mcq":
            return format_nextqa_mcq(data_list, self.answer_flag)
        elif self.dataset_name == "mlvu-mcq":
            return format_mlvu_mcq(data_list, self.answer_flag)
        else:
            raise NotImplementedError(f"{self.dataset_name} Formating for VideoQAEvalLoader is not yet implemented")
    
    def __len__(self):
        return len(self.data_list)

    def __iter__(self):
        """Make the loader iterable over its data list."""
        for idx in range(len(self)):
            yield self.__call__(idx)

    def __call__(self, idx):
        data = self.data_list[idx]
        qid = data['qid']
        vid = data['vid']
        conversations = copy.deepcopy(data['conversations'])
        video_file = os.path.join(self.video_root, data['video_filepath'])

        ## read video
        frames, video_time, frame_time, num_frames = self.get_frames(video_file)

        ## preprocess instruction
        if self.data_args.add_time_instruction:
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
            conversations = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{conversations.replace(DEFAULT_IMAGE_TOKEN, "")}'

        ## tokenizing instruction
        conv = self.tokenizer.conv_tmpl.copy()
        conv.append_message(conv.roles[0], conversations)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_length = get_prompt_stat(input_ids, IMAGE_TOKEN_INDEX)

        ## formating into dict
        if isinstance(idx, int):
            data_dict = dict(input_ids=input_ids)
        data_dict["image"] = frames
        data_dict["modality"] = "video"
        data_dict["id"] = qid
        data_dict["vid"] = vid
        data_dict["sys_len"] = prompt_length['sys']
        data_dict["inst_len"] = prompt_length['inst']
        data_dict["frame_len"] = len(frames)
        data_dict["instruction"] = conversations
        data_dict["answer"] = str(data['answer']) if self.answer_flag else ""

        return data_dict

    def toggle_memory(self, video_file):
        """Adjust memory usage based on the access pattern - reset to counter 0 if exist else increase it"""
        curr_keys = list(self.memory_counter.keys())
        for k in curr_keys:
            if k != video_file:
                self.memory_counter[k] += 1
            else:
                self.memory_counter[k] = 0

            if self.memory_counter[k] == self.memory_thresh:
                self.remove_memory(k)

    def remove_memory(self, k):
        del self.memory_frames[k]
        del self.memory_video_time[k]
        del self.memory_frame_time[k]
        del self.memory_num_frames[k]
        del self.memory_counter[k]
        gc.collect()
        # print(f"[Memory Remove] {k}")

    def add_memory(self, video_file, video_frames, video_time, frame_time, num_frames):
        """Store new data - first time loaded."""
        self.memory_frames[video_file] = video_frames
        self.memory_video_time[video_file] = video_time
        self.memory_frame_time[video_file] = frame_time
        self.memory_num_frames[video_file] = num_frames
        self.memory_counter[video_file] = 0
        # print(f"[Memory Add] {video_file}")

    def get_frames(self, video_file):
        """Retrieve frames from memory or load them if not present."""
        if video_file in self.memory_frames:
            # print(f"[Memory Access] {video_file}")
            video_frames, video_time, frame_time, num_frames = self.memory_frames[video_file], self.memory_video_time[video_file], self.memory_frame_time[video_file], self.memory_num_frames[video_file]
            self.toggle_memory(video_file)
        else:
            video_frames, video_time, frame_time, num_frames = process_video_with_decord(video_file, self.data_args)
            video_frames = self.data_args.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
            self.add_memory(video_file, video_frames, video_time, frame_time, num_frames)
            self.toggle_memory(video_file)

        return video_frames, video_time, frame_time, num_frames

    def get_raw_frames(self, idx=-1, video_file=None):
        """Return RGB frames for visualization"""
        get_by_idx = idx > -1
        if get_by_idx:
            data = self.data_list[idx]
            video_file = os.path.join(self.video_root, data['video_filepath'])
        video_frames, _, _, _ = process_video_with_decord(video_file, self.data_args)

        return video_frames


class VidQA_Loader_Feature(Dataset):
    def __init__(self, dataset_name: str, anno_filepath: str, data_root: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, 
            answer_flag: bool = False, prev_pred_qids=None, first_sample=False):
        
        self.dataset_name = dataset_name
        self.feature_dir = os.path.join(data_root, "features")
        self.metadata_dir = os.path.join(data_root, "metadata")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.answer_flag = answer_flag
        data_list = read_json(anno_filepath) # suppose anno_filepath in jsonl format     

        if len(data_list) == 0:
            self.data_list = []
        else:
            self.data_list = self.format_data(data_list)

        if first_sample: # for debugging purpose
            self.data_list = [self.data_list[0]]
        
        if prev_pred_qids is not None:
            new_data_list = []
            for d in self.data_list:
                if d['qid'] not in prev_pred_qids:
                    new_data_list.append(d)
        
            self.data_list = new_data_list
        rank0_print(f"Loading {anno_filepath}... {len(self.data_list)} samples ")

        self.temporal_skip_freq = 1
        # self.temporal_skip_freq = 4 # for debugging with shorter sequence

    def format_data(self, data_list):
        if self.dataset_name == "videomme":
            return format_videomme(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench_short":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "egoschema":
            return format_egoschema(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-val":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-test":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "nextqa-mcq":
            return format_nextqa_mcq(data_list, self.answer_flag)
        elif self.dataset_name == "mlvu-mcq":
            return format_mlvu_mcq(data_list, self.answer_flag)
        else:
            raise NotImplementedError(f"{self.dataset_name} Formating for VideoQAEvalLoader is not yet implemented")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        qid = data['qid']
        vid = data['vid']
        conversations = copy.deepcopy(data['conversations'])
        
        ## read preprocessed data
        feature_filepath = os.path.join(self.feature_dir, f"{vid}.pt")
        metadata_filepath = os.path.join(self.metadata_dir, f"{vid}.pkl")

        with open(metadata_filepath, "rb") as fp:
            metadata = pickle.load(fp)
        video_time = metadata['video_time']
        frame_time = metadata['frame_time']
        num_frames = metadata['num_frames']
        video_feature = torch.load(feature_filepath, weights_only=True)

        if self.temporal_skip_freq > 2:
            video_feature = video_feature[0::self.temporal_skip_freq]
        ## preprocess instruction
        if self.data_args.add_time_instruction:
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
            conversations = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{conversations.replace(DEFAULT_IMAGE_TOKEN, "")}'
        else:
            conversations = f'{DEFAULT_IMAGE_TOKEN}\n{conversations.replace(DEFAULT_IMAGE_TOKEN, "")}'
        
        ## tokenizing instruction
        conv = self.tokenizer.conv_tmpl.copy()
        conv.append_message(conv.roles[0], conversations)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        prompt_length = get_prompt_stat(input_ids, IMAGE_TOKEN_INDEX)

        ## formating into dict
        if isinstance(idx, int):
            data_dict = dict(input_ids=input_ids)
        data_dict["feature"] = video_feature 
        data_dict["modality"] = "video_feature"
        data_dict["id"] = qid
        data_dict["vid"] = vid
        data_dict["sys_len"] = prompt_length['sys']
        data_dict["inst_len"] = prompt_length['inst']
        data_dict["frame_len"] = len(video_feature)
        data_dict["answer"] = str(data['answer']) if self.answer_flag else ""
        data_dict["instruction"] = conversations

        return data_dict


class VidQA_Loader_Feature_Qwen2VL(Dataset):
    def __init__(self, dataset_name: str, anno_filepath: str, data_root: str, data_args: DataArguments, 
            answer_flag: bool = False, prev_pred_qids=None, first_sample=False):
        
        self.dataset_name = dataset_name
        self.feature_dir = os.path.join(data_root, "features")
        self.metadata_dir = os.path.join(data_root, "metadata")
        self.data_args = data_args
        self.answer_flag = answer_flag
        data_list = read_json(anno_filepath) # suppose anno_filepath in jsonl format     

        if len(data_list) == 0:
            self.data_list = []
        else:
            self.data_list = self.format_data(data_list)

        if first_sample: # for debugging purpose
            self.data_list = [self.data_list[0]]
        
        if prev_pred_qids is not None:
            new_data_list = []
            for d in self.data_list:
                if d['qid'] not in prev_pred_qids:
                    new_data_list.append(d)
        
            self.data_list = new_data_list
        rank0_print(f"Loading {anno_filepath}... {len(self.data_list)} samples ")

    def format_data(self, data_list):
        if self.dataset_name == "videomme":
            return format_videomme(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "vnbench_short":
            return format_vnbench(data_list, self.answer_flag)
        elif self.dataset_name == "egoschema":
            return format_egoschema(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-val":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "lvb-test":
            return format_lvb(data_list, self.answer_flag)
        elif self.dataset_name == "nextqa-mcq":
            return format_nextqa_mcq(data_list, self.answer_flag)
        elif self.dataset_name == "mlvu-mcq":
            return format_mlvu_mcq(data_list, self.answer_flag)
        else:
            raise NotImplementedError(f"{self.dataset_name} Formating for VideoQAEvalLoader is not yet implemented")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        qid = data['qid']
        vid = data['vid']
        conversations = copy.deepcopy(data['conversations'])
        
        ## read preprocessed data
        feature_filepath = os.path.join(self.feature_dir, f"{vid}.pt")
        metadata_filepath = os.path.join(self.metadata_dir, f"{vid}.pkl")

        with open(metadata_filepath, "rb") as fp:
            metadata = pickle.load(fp)
        video_sample_fps = metadata['video_sample_fps']
        frame_time = metadata['frame_time']
        video_grid_thw = metadata['video_grid_thw']
        video_feature = torch.load(feature_filepath, weights_only=True)

        ## tokenizing instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": feature_filepath},
                    {"type": "text", "text": conversations},
                ],
            }
        ]
        text = [self.data_args.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        merge_length = self.data_args.processor.image_processor.merge_size**2
        index = 0
        for i in range(len(text)):
            while "<|video_pad|>" in text[i]:
                num_video_tokens = video_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace("<|video_pad|>", "<|placeholder|>" * num_video_tokens, 1)
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")
        text_inputs = self.data_args.processor.tokenizer(text, padding=True, return_tensors="pt")
        input_ids = text_inputs['input_ids'][0]
        attention_mask = text_inputs['attention_mask'][0]

        video_token_id = self.data_args.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        prompt_length = get_prompt_stat(input_ids, video_token_id)

        ## formating into dict
        data_dict = {"input_ids": input_ids, "attention_mask": attention_mask, 
                    "feature": video_feature, "video_grid_thw": video_grid_thw, "video_sample_fps": video_sample_fps}
        data_dict["id"] = qid
        data_dict["vid"] = vid
        data_dict["sys_len"] = prompt_length['sys']
        data_dict["inst_len"] = prompt_length['inst']
        data_dict["frame_len"] = len(video_feature)
        data_dict["answer"] = str(data['answer']) if self.answer_flag else ""
        data_dict["instruction"] = conversations

        return data_dict