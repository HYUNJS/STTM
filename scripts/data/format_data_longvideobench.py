import json
import os
import string
import pandas as pd

def read_json(filepath):
    with open(filepath, "r") as fp:
        data = json.load(fp)
    return data

def convert_anno_v2(df, new_anno_filepath):
    """
        Only necessary information w/o subtitle
    """
    # df = df_val
    # new_anno_filepath = new_anno_filepath_val
    df_new = pd.DataFrame()
    
    df_new['videoID'] = df['video_id']
    df_new['question_id'] = df['id']
    df_new['duration'] = df['duration_group']
    df_new['task_type'] = df['question_category']
    df_new['question'] = df['question']
    df_new['options'] = df['candidates']
    if "correct_choice" in df:
        df_new['answer'] = df['correct_choice']
    
    ## handling edge case
    edge_case_mask = df['video_id'].apply(lambda x: "@" in x)
    df_new.loc[edge_case_mask, 'videoID'] = df_new.loc[edge_case_mask, 'videoID'].str.split('-').str[-1]
    df_new.loc[edge_case_mask, 'question_id'] = df_new.loc[edge_case_mask, 'question_id'].str.split('-').str[-1]
    
    results = df_new.to_dict('records')
    with open(new_anno_filepath, 'w') as fp:
        json.dump(results, fp, indent=2)

if __name__ == "__main__":
    data_root = "datasets/longvideobench/annotations"
    anno_filepath_test = os.path.join(data_root, "lvb_test_wo_gt.json")
    anno_filepath_val = os.path.join(data_root, "lvb_val.json")
    new_anno_filepath_test = os.path.join(data_root, "lvb_test_v2.json")
    new_anno_filepath_val = os.path.join(data_root, "lvb_val_v2.json")
    annos_val = read_json(anno_filepath_val)
    annos_test = read_json(anno_filepath_test)

    df_val = pd.DataFrame(annos_val)
    df_test = pd.DataFrame(annos_test)
    df_val
    # convert_anno_v2(df_val, new_anno_filepath_val)
    # convert_anno_v2(df_test, new_anno_filepath_test)
