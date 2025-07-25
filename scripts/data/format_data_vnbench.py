import json
import os
import string
import pandas as pd
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
    Download data from
    https://drive.google.com/file/d/1KOUzy07viQzpmpcBqydUA043VQZ4nmRv/view
    """

    # data_root = "datasets/VideoNIAH_data/vnbench"
    data_root = "datasets/vnbench/annotations"
    anno_filepath = os.path.join(data_root, "VNBench-main-4try.json")
    new_anno_filepath = os.path.join(data_root, "VNBench-main-4try_v2.json")
    with open(anno_filepath, "r") as fp:
        annos = json.load(fp)

    df = pd.DataFrame(annos)

    df['videoID'] = df["video"].apply(lambda x: x.split("/")[-1].replace(".mp4", ""))
    df['question_id'] = df.apply(lambda x: f"{x['videoID']}_try{x['try']}", axis=1)

    prefixes = list(string.ascii_uppercase)  # ['A', 'B', 'C', ...]
    df['options'] = df['options'].apply(lambda x: [f"{prefixes[i]}. {x[i]}." for i in range(len(x))])

    df['video_second'] = df['videoID'].apply(lambda x: get_video_second(x))
    df['duration'] = df['video_second'].apply(lambda x: get_video_duration_category(x))

    df = df.drop(["video", "gt"], axis=1)
    df = df.rename(columns={"gt_option": "answer", "type": "task_type"})

    results = df.to_dict('records')
    with open(new_anno_filepath, 'w') as fp:
        json.dump(results, fp, indent=2)

    with open(new_anno_filepath, 'r') as fp:
        annos = json.load(fp)
