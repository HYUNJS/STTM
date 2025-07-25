import os
import shutil
import decord
import pandas as pd
import numpy as np
import os.path as osp
import json
from tqdm import tqdm


def get_answer_index(row):
    try:
        return row['candidates'].index(row['answer'])
    except ValueError:
        return -1  # or None, if you prefer

def get_video_second(x):
    vr = decord.VideoReader(os.path.join(data_root, "videos", f"{x}.mp4"))
    return len(vr) / vr.get_avg_fps()

def get_video_duration_category(x):
    if x < 60:
        return "short"
    elif x < 120:
        return "medium"
    else:
        return "long"

def reorganize_videos(data_root, tgt_type):
    src_video_root = osp.join(data_root, "video", tgt_type)
    src_filenames = os.listdir(src_video_root)
    for src_filename in src_filenames:
        src_video_filepath = osp.join(src_video_root, src_filename)
        tgt_filename = f"{tgt_type}-{src_filename}"
        tgt_video_filepath = osp.join(data_root, "videos", tgt_filename)
        shutil.move(src_video_filepath, tgt_video_filepath)

def reformat_annotations(tgt_type):
    anno_filepath = osp.join(data_root, "json", f"{tgt_type}.json")
    new_anno_filepath = osp.join(data_root, "annotations", f"{tgt_type}_v2.json")
    with open(anno_filepath, "r") as fp:
        annos = json.load(fp)
    ## Reformat columns
    df = pd.DataFrame(annos)
    df['video_id'] = df['video'].apply(lambda x: f"{tgt_type}-"+x.replace(".mp4", ""))
    df['question_idx'] = df.groupby('video_id').cumcount() + 1
    df['question_id'] = df['video_id'] + '-' + df['question_idx'].astype(str)
    df['answer_index'] = df.apply(get_answer_index, axis=1)
    assert (df['answer_index'] == -1).sum() == 0, f"Reformating {tgt_type} faces unexpected format"
    df['answer_in_letter'] = df['answer_index'].apply(lambda x: chr(ord("A")+x))
    df['video_second'] = df['video_id'].apply(lambda x: get_video_second(x))
    df['duration'] = df['video_second'].apply(lambda x: get_video_duration_category(x))
    
    ## Save in new annotation file
    vids = df['video_id'].tolist()
    qids = df['question_id'].tolist()
    questions = df['question'].tolist()
    answers = df['answer_in_letter'].tolist()
    options = df['candidates'].tolist()
    task_types = df['question_type'].tolist()
    durations = df['duration'].tolist()
    video_seconds = df['video_second'].tolist()

    new_anno_df = pd.DataFrame.from_dict({
        "video_id": vids,
        "question_id": qids,
        "question": questions,
        "answer": answers,
        "options": options,
        "task_type": task_types,
        "duration": durations,
        "video_second": video_seconds,
    })
    new_anno_data = new_anno_df.to_dict(orient='records')
    
    with open(osp.join(data_root, new_anno_filepath), "w") as fp:
        json.dump(new_anno_data, fp)
    
if __name__ == "__main__":
    """
    Download data from https://huggingface.co/datasets/MLVU/MVLU
    """
    
    mcq_type_list = ["1_plotQA", "2_needle", "3_ego", "4_count", "5_order", "6_anomaly_reco", "7_topic_reasoning"]
    gen_type_list = ["8_sub_scene", "9_summary"]
    data_root = "datasets/mlvu"
    
    ## Please firstly reorganize video before reformating annotations
    for tgt_type in tqdm([*mcq_type_list, *gen_type_list]):
        reorganize_videos(data_root, tgt_type)

    ## save reformated annotations
    for tgt_type in tqdm(mcq_type_list):
        reformat_annotations(tgt_type)
    
    merged_annos = []
    for tgt_type in tqdm(mcq_type_list):
        anno_filepath = osp.join(data_root, "annotations", f"{tgt_type}_v2.json")
        with open(anno_filepath, "r") as fp:
            annos = json.load(fp)
        merged_annos.extend(annos)
    
    merged_anno_filepath = osp.join(data_root, "annotations", f"MLVU_mcq_v2.json")
    with open(osp.join(data_root, merged_anno_filepath), "w") as fp:
        json.dump(merged_annos, fp)
    