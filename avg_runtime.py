import pickle
import os
import numpy as np
import pandas as pd
import json
import argparse


def print_stat(tgt_df, group, ttft_per_group):
    ttft_llm = tgt_df['ttft_llm'].mean()
    time_decoding = tgt_df['time_decoding'].mean()
    num_dec_token = tgt_df['num_dec_token'].mean()
    time_all = tgt_df['all'].mean()
    ttft_per_group[group] = ttft_llm
    stats = {"ttft_llm": ttft_llm, "time_dec": time_decoding, "time_all": time_all, "token_dec": num_dec_token}
    print(f"{group}")
    print(pd.DataFrame([stats]).to_csv(index=False, sep=","))
    return stats

def measure_runtime_with_duration_info(pred_filepath, gt_filepath, save_flag=False):
    print(pred_filepath)
    os.makedirs(os.path.dirname(pred_filepath.replace("outputs", "metrics")), exist_ok=True)
    with open(gt_filepath, "r", encoding='utf-8') as f:
        gt = json.load(f)
    with open(pred_filepath, "rb") as fp:
        runtimes = pickle.load(fp)

    runtimes = pd.DataFrame(runtimes).sort_values(by="qid").reset_index(drop=True)
    runtimes = pd.merge(runtimes, pd.DataFrame(gt)[["question_id", "duration"]], left_on="qid", right_on="question_id")

    ttft_per_group = {}
    overall_stats = print_stat(runtimes, "overall", ttft_per_group)
    runtimes_gb = runtimes.groupby("duration")
    for k in ['short', 'medium', 'long']: # for video-mme
    # for k in ['long']: # for video-mme long video only
        if k not in runtimes_gb.groups:
            ttft_per_group[k] = -1.0
        else:
            print_stat(runtimes_gb.get_group(k), k, ttft_per_group)

    overall_stats.update(ttft_per_group)
    overall_stats.pop("overall")
    overall_stats_df = pd.DataFrame([overall_stats])
    print(overall_stats_df.to_csv(index=False, sep=","))
    if save_flag:
        metric_filepath = pred_filepath.replace("outputs", "metrics").replace(".pkl", ".csv")
        overall_stats_df.to_csv(metric_filepath, index=False)

def measure_runtime_no_duration_info(pred_filepath, save_flag=False):
    print(pred_filepath)
    os.makedirs(os.path.dirname(pred_filepath.replace("outputs", "metrics")), exist_ok=True)
    with open(pred_filepath, "rb") as fp:
        runtimes = pickle.load(fp)

    runtimes = pd.DataFrame(runtimes).sort_values(by="qid").reset_index(drop=True)
    ttft_per_group = {}
    overall_stats = print_stat(runtimes, "overall", ttft_per_group)

    overall_stats.update(ttft_per_group)
    overall_stats.pop("overall")
    overall_stats_df = pd.DataFrame([overall_stats])
    print(overall_stats_df.to_csv(index=False, sep=","))
    if save_flag:
        metric_filepath = pred_filepath.replace("outputs", "metrics").replace(".pkl", ".csv")
        overall_stats_df.to_csv(metric_filepath, index=False)

def measure_runtime(pred_filepath, gt_filepath=None, save_flag=False):
    if gt_filepath is None or "egoschema" in gt_filepath:
        measure_runtime_no_duration_info(pred_filepath, save_flag)
    else:
        measure_runtime_with_duration_info(pred_filepath, gt_filepath, save_flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_filepath", type=str, required=True)
    parser.add_argument("--gt_filepath", type=str, default="datasets/videomme/annotations/videomme.json")
    # parser.add_argument("--gt_filepath", type=str, default="datasets/vnbench/annotations/VNBench-main-4try_v2.json")
    # parser.add_argument("--gt_filepath", type=str, default="datasets/egoschema/annotations/questions.json")
    args = parser.parse_args()
    anno_filepath = args.gt_filepath
    tgt_filepath = args.pred_filepath

    measure_runtime(tgt_filepath, anno_filepath, save_flag=True)
