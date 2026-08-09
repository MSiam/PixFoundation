import argparse
import pandas as pd
import os
import json
import copy
from tqdm import tqdm
import numpy as np
import random
import string
from datasets import load_dataset
from pixf_utils.prompt_sens_utils  import alter_text_util

def alter_text(text, N):
    error_type = np.random.randint(3)
    return alter_text_util(text, N, error_type)

def change_mask_box(original_template):
    box_template = original_template.replace('mask', 'box')
    box_template = box_template.replace('masks', 'boxes')
    box_template = box_template.replace('segment', 'detect')
    return box_template

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--n_variations_per_prompt", type=int, default=2)
    parser.add_argument("--source_file", type=str) # Prompt templates file in full prompt, objects file in expression only
    parser.add_argument("--stop_at", type=int, default=29)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--variations_type", type=str, default="full_prompt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.variations_type == "expression_only":
        df_objects = pd.read_csv(args.source_file, index_col=0)
        df_objects_output = copy.deepcopy(df_objects)

        for index, row in tqdm(df_objects.iterrows()):
            Object = row[' Object']
            tokens = Object.strip().split(' ')
            modified_object = ' '
            for token in tokens:
                try:
                    modified_token = alter_text(token, 1)
                except:
                    modified_token = token
                modified_object += modified_token + ' '

            df_objects_output[' Object'].iloc[index-1] = modified_object
        df_objects_output.to_csv(os.path.join(args.root, 'Objects_variations.csv'))

    elif args.variations_type == "full_prompt":
        templates_f = open(args.source_file, 'r')
        templates = pd.read_csv(templates_f)

        ntemplates= len(templates['Template'])
        var_df_dict_arr = []
        var_df_dict_arr_box = []

        n_chars_to_alter = 8

        for var_i in range(args.n_variations_per_prompt*ntemplates):

            if var_i > args.stop_at:
                break
            if var_i < args.start_at:
                continue

            original_template = templates['Template'][var_i%ntemplates]
            obj_idx = original_template.index('{}')
            nstart = int((obj_idx/(len(original_template)-4))*n_chars_to_alter)

            modified_template = alter_text(original_template[:obj_idx], nstart)  + '{}' + alter_text(original_template[obj_idx+2:],
                                                                                                     n_chars_to_alter - nstart)

            box_template = change_mask_box(original_template)
            obj_idx = box_template.index('{}')
            modified_template_box = alter_text(box_template[:obj_idx], nstart)  + '{}'
            modified_template_box += alter_text(box_template[obj_idx+2:], n_chars_to_alter - nstart)

            var_df_dict_arr.append({'Index': var_i, 'Template': modified_template})
            var_df_dict_arr_box.append({'Index': var_i, 'Template': modified_template_box})

        var_df = pd.DataFrame(var_df_dict_arr)
        var_df = var_df.set_index('Index')
        var_df.to_csv(os.path.join(args.root, 'lang_grounding_templates.csv'))

        var_df_boxes = pd.DataFrame(var_df_dict_arr_box)
        var_df_boxes = var_df_boxes.set_index('Index')
        var_df_boxes.to_csv(os.path.join(args.root, 'lang_grounding_templates_box.csv'))
