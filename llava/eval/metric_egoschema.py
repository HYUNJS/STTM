import os
import argparse
import json
import re
import pandas as pd
import requests


def read_json(tgt_filepath):
    with open(tgt_filepath, 'r') as fp:
        data = json.load(fp)
    return data

def send_post_request(json_file):
    """
    Sends a POST request to the specified URL with the given JSON file.

    Parameters:
    - json_file (str): Path to the JSON file to be used in the request body.

    Returns:
    - Response object containing server's response.
    """

    url = "https://validation-server.onrender.com/api/upload/"
    headers = {
        "Content-Type": "application/json"
    }

    with open(json_file, 'r') as f:
        data = json.load(f)

    response = requests.post(url, headers=headers, json=data)
    
    return response

def extract_characters_regex_egoschema(s):
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
    match_letter = matches[0]
    letter_to_digit = {
        "A": "0",
        "B": "1",
        "C": "2",
        "D": "3",
        "E": "4",
    }
    digit = letter_to_digit.get(match_letter, "")
    return digit

def save_submission_ver(pred_filepath):
    new_pred_filepath = pred_filepath.replace(".json", "_submission.json")
    if os.path.isfile(new_pred_filepath):
        return new_pred_filepath
    
    pred = read_json(pred_filepath)
    pred_df = pd.DataFrame(pred)
    qid = pred_df['question_id']
    digit_answer = pred_df['answer'].apply(lambda x: extract_characters_regex_egoschema(x))
    new_pred_df = pd.concat([qid, digit_answer], axis=1)
    new_pred = new_pred_df.set_index('question_id')['answer'].to_dict()

    with open(new_pred_filepath, "w") as fp:
        json.dump(new_pred, fp)
    return new_pred_filepath

def get_acc(pred_filepath):
    response = send_post_request(pred_filepath)
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content:\n{response.text}")
    acc_test, acc_subset = -1.0, -1.0
    if response.status_code == 200:
        response_by_line = response.text.split("\n")
        acc_test = float(re.search(r'accuracy:\s*([\d.]+)', response_by_line[1]).group(1)) * 100
        acc_subset = float(re.search(r'accuracy:\s*([\d.]+)', response_by_line[2]).group(1)) * 100
    return acc_test, acc_subset

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

def evaluate(pred_filepath, gt_filepath=None):
    pred_model_name = os.path.basename(pred_filepath).replace(".json", "")
    pred_param_dict = parse_params_from_filename(pred_model_name)
    metric_filepath = pred_filepath.replace("outputs", "metrics").replace(".json", "_accuracy.csv")
    os.makedirs(os.path.dirname(metric_filepath), exist_ok=True)
    
    ## Compute accuracy
    subm_filepath = save_submission_ver(pred_filepath)
    acc_test, acc_subset = get_acc(subm_filepath)
    
    ## Save result
    metric = pd.DataFrame({"model": [pred_model_name], **pred_param_dict, 'total': [acc_test], "subset": [acc_subset]}, index=[0]).reset_index(drop=True)
    metric.to_csv(metric_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_filepath", type=str, required=True)
    args = parser.parse_args()

    pred_filepath = args.pred_filepath
    evaluate(pred_filepath)
    
