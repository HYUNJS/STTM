import os
import argparse
import json
import re
import pandas as pd


DURATION_CATEGORIES = ["short", "medium", "long"]
TASK_CATEGORIES = ['CH', 'CW', 'DC', 'DL', 'DO', 'TC', 'TN', 'TP']


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    if len(s.split()) > 10 and not re.search("[ABCDE]", s):
        return ""
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        return ""
    return matches[0]

def get_acc(tgt_df):
    return 100 * tgt_df['correct'].sum() / len(tgt_df) if len(tgt_df) > 0 else -1

def read_json(tgt_filepath):
    with open(tgt_filepath, 'r') as fp:
        data = json.load(fp)
    return data

def compute_accs(tgt_df, tgt_group_dict):
    total_acc = get_acc(tgt_df)
    per_group_accs = {}
    for tgt_group_name, tgt_group_value in tgt_group_dict.items():
        tgt_group_accs = {}
        for tgt_type in tgt_group_value:
            tgt_group_df = tgt_df[tgt_df[tgt_group_name] == tgt_type]
            tgt_acc = get_acc(tgt_group_df)
            tgt_group_accs[tgt_type] = tgt_acc
        per_group_accs[tgt_group_name] = tgt_group_accs
    
    return total_acc, per_group_accs

def print_accs(total_acc, per_group_accs, print_per_group=False):
    ## Print accuracy
    print(f"Total Acc: {total_acc:.1f}")
    if not print_per_group:
        return
    
    for tgt_group_name in per_group_accs:
        tgt_group_acc_log = "="*50
        tgt_group_acc_log += f"\n[{tgt_group_name}]\n"
        tgt_group_accs = per_group_accs[tgt_group_name]
        for k in tgt_group_accs:
            tgt_group_acc_log += f"{k}: {tgt_group_accs[k]:.1f}\n"
        print(tgt_group_acc_log)

def parse_params_from_filename(pred_model_name): ## change groupwise metrics for vnbench
    start_pos = pred_model_name.find("_F-")
    if start_pos == -1:
        print("Unexpected filename pattern")
        raise NotImplementedError("Parsing of this input filename is not expected")
    
    pred_model_name = pred_model_name[start_pos:]
    param_list = pred_model_name.split("_")
    param_dict = {
        "frames test": 0,
        "fps test": 0,
        "2d-pool": 0,
        "slowfast": 0,
        "2d-pool slow": 0,
        "stride slow": 0,
        "instr": 0,
    }
    ## common params
    param_dict["frames test"] = param_list[1].split("-")[1]
    param_dict["fps test"] = param_list[2].split("-")[1]
    if len(param_list) > 3:
        param_dict["2d-pool"] = param_list[3].split("-")[1]

    if len(param_list) > 4:
        for param in param_list[4:]:
            ## check slowfast
            if "SF-" in param:
                _, pool_slow, stride_slow = param.split("-")
                param_dict["slowfast"] = 1
                param_dict["2d-pool slow"] = int(pool_slow.replace("s", ""))
                param_dict["stride slow"] = int(stride_slow.replace("t", ""))

            ## instruction-aware
            if "instr-" in param:
                param_dict["instr"] = int(param.split("-")[1])
    
    param_dict = {k: [v] for k, v in param_dict.items()}

    return param_dict


def evaluate(pred_filepath, gt_filepath):
    pred_model_name = os.path.basename(pred_filepath).replace(".json", "")
    pred_param_dict = parse_params_from_filename(pred_model_name)
    metric_filepath = pred_filepath.replace("outputs", "metrics").replace(".json", "_accuracy.csv")
    os.makedirs(os.path.dirname(metric_filepath), exist_ok=True)
    
    ## Read data
    pred_list, gt_list = read_json(pred_filepath), read_json(gt_filepath)
    for pred in pred_list:
        parsed_ans = extract_characters_regex(pred['answer'])
        pred['parsed_answer'] = parsed_ans
        pred['correct'] = pred['gt'] == parsed_ans
        pred['answered'] = parsed_ans != "" # count valid answer 

    pred_df = pd.DataFrame(pred_list)
    gt_df = pd.DataFrame(gt_list)[['question_id', 'video_id', 'duration', 'task_type']]
    merged_df = pd.merge(gt_df, pred_df, on='question_id')
    
    ## Compute accuracy
    tgt_df = merged_df
    tgt_group_dict = {
                    "duration": DURATION_CATEGORIES,
                    "task_type": TASK_CATEGORIES,
                }
    total_acc, per_group_accs = compute_accs(tgt_df, tgt_group_dict)
    print_accs(total_acc, per_group_accs)

    ## Save result
    df1 = pd.DataFrame({"model": [pred_model_name], **pred_param_dict, 'total': [total_acc]}, index=[0]).reset_index(drop=True)
    df2 = pd.DataFrame.from_dict(per_group_accs['duration'], orient='index', columns=[0]).transpose().reset_index(drop=True)
    df3 = pd.DataFrame.from_dict(per_group_accs['task_type'], orient='index', columns=[0]).transpose().reset_index(drop=True)
    metric_all = pd.concat([df1, df2, df3], axis='columns')
    metric_all.to_csv(metric_filepath, index=False)

def evaluate_per_duration_group(pred_filepath, gt_filepath):
    pred_model_name = os.path.basename(pred_filepath).replace(".json", "")
    pred_param_dict = parse_params_from_filename(pred_model_name)
    metric_filepath = pred_filepath.replace("outputs", "metrics_per_duration").replace(".json", ".csv")
    os.makedirs(os.path.dirname(metric_filepath), exist_ok=True)
    
    ## Read data
    pred_list, gt_list = read_json(pred_filepath), read_json(gt_filepath)
    for pred in pred_list:
        parsed_ans = extract_characters_regex(pred['answer'])
        pred['parsed_answer'] = parsed_ans
        pred['correct'] = pred['gt'] == parsed_ans
        pred['answered'] = parsed_ans != "" # count valid answer 

    pred_df = pd.DataFrame(pred_list)
    gt_df = pd.DataFrame(gt_list)[['question_id', 'video_id', 'duration', 'task_type']]
    merged_df = pd.merge(gt_df, pred_df, on='question_id')
    merged_df_gb = merged_df.groupby("duration")

    tgt_group_dict = {
                    "duration": DURATION_CATEGORIES,
                    "task_type": TASK_CATEGORIES,
                }
    metric_all_group = []
    for tgt_duration in DURATION_CATEGORIES:
        print(f"Target Duration {tgt_duration}")
        tgt_df = merged_df_gb.get_group(tgt_duration)
        total_acc, per_group_accs = compute_accs(tgt_df, tgt_group_dict)
        print_accs(total_acc, per_group_accs)

        df1 = pd.DataFrame({"model": [pred_model_name], **pred_param_dict, "duration": [tgt_duration], 'total': [total_acc]}, index=[0]).reset_index(drop=True)
        df2 = pd.DataFrame.from_dict(per_group_accs['task_type'], orient='index', columns=[0]).transpose().reset_index(drop=True)
        tgt_metric_all = pd.concat([df1, df2], axis='columns')
        metric_all_group.append(tgt_metric_all)

    metric_all = pd.concat(metric_all_group)
    metric_all.to_csv(metric_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_filepath", type=str, required=True)
    parser.add_argument("--gt_filepath", type=str, default="datasets/nextqa/annotations/MC_test_v2.json")
    parser.add_argument("--per_duration", action="store_true", default=False)
    args = parser.parse_args()

    pred_filepath = args.pred_filepath
    gt_filepath = args.gt_filepath
    per_duration_flag = args.per_duration
    if per_duration_flag:
        evaluate_per_duration_group(pred_filepath, gt_filepath)
    else:
        evaluate(pred_filepath, gt_filepath)
    
