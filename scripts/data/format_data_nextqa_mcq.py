import json
import os
import os.path as osp
import pandas as pd
import numpy as np
import decord

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


if __name__ == "__main__":
    """
    Download the data from https://huggingface.co/datasets/lmms-lab/NExTQA
    which provides he resized and id-mapped videos
    """
    data_root = "datasets/nextqa"
    mcq_test_anno_filename = "MC/test-00000-of-00001.parquet"
    new_mcq_test_anno_filename = "annotations/MC_test_v2.json"
    mcq_test_anno_filepath = osp.join(data_root, mcq_test_anno_filename)
    mcq_anno_data = pd.read_parquet(mcq_test_anno_filepath)

    mcq_anno_data['video_id'] = mcq_anno_data['video'].apply(lambda x: str(x))
    mcq_anno_data['question_id'] = mcq_anno_data.apply(lambda row: f"{row['video']}_{row['qid']}", axis=1)
    option_cols = ['a0', 'a1', 'a2', 'a3', 'a4']
    mcq_anno_data['options'] = mcq_anno_data[option_cols].apply(lambda row: list(row), axis=1)
    mcq_anno_data['answer_in_letter'] = mcq_anno_data['answer'].apply(lambda x: chr(ord("A")+x))
    mcq_anno_data['video_second'] = mcq_anno_data['video'].apply(lambda x: get_video_second(x))
    mcq_anno_data['duration'] = mcq_anno_data['video_second'].apply(lambda x: get_video_duration_category(x))
    
    vids = mcq_anno_data['video_id']
    qids = mcq_anno_data['question_id']
    questions = mcq_anno_data['question']
    answers = mcq_anno_data['answer_in_letter']
    options = mcq_anno_data['options']
    task_types = mcq_anno_data['type']
    durations = mcq_anno_data['duration']
    video_seconds = mcq_anno_data['video_second']

    new_mc_anno_df = pd.DataFrame.from_dict({
        "video_id": vids.to_numpy(),
        "question_id": qids.to_numpy(),
        "question": questions.to_numpy(),
        "answer": answers.to_numpy(),
        "options": options.to_numpy(),
        "task_type": task_types.to_numpy(),
        "duration": durations.to_numpy(),
        "video_second": video_seconds.to_numpy(),
    })
    new_mc_anno_data = new_mc_anno_df.to_dict(orient='records')
    
    with open(osp.join(data_root, new_mcq_test_anno_filename), "w") as fp:
        json.dump(new_mc_anno_data, fp)