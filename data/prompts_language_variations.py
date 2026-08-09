import argparse
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import random
import string
from datasets import load_dataset
from pixf_utils.prompt_sens_utils  import alter_text_util

def alter_text(text, N):
    error_type = np.random.randint(3)
    return alter_text_util(text, N, error_type)

def generate_variation(variation_type, paraphrase, template, original_question, original_choices,
                       question_extension, dataset_type):
    # variation type: 1 - spelling errors, 2: prompt templates, 3: paraphrasing
    if variation_type == 1:
        n_chars_to_alter = 7
        question = alter_text(original_question, n_chars_to_alter)
        question_extension = alter_text(question_extension, n_chars_to_alter)

        #choices = original_choices
        if dataset_type == 'mmvp':
            choices = original_choices.split('(b)')
            choices[1] = '(b)' + choices[1]
            choices = [part.strip() for part in choices]
        elif dataset_type == 'cvbench':
            choices = original_choices

        for i, part in enumerate(choices):
            try:
                choices[i] = part[:4] + alter_text(part[4:], 1)
            except:
                continue
        choices = ' '.join(choices)
    elif variation_type == 2:
        n_chars_to_alter = 7
        question_extension = alter_text(question_extension, n_chars_to_alter)

        if dataset_type == 'mmvp':
            choices = original_choices.split('(b)')
            choices[1] = '(b)' + choices[1]
            choices = [part.strip() for part in choices]
        elif dataset_type == 'cvbench':
            choices = original_choices


        for i, part in enumerate(choices):
            try:
                choices[i] = part[:4] + alter_text(part[4:], 1)
            except:
                continue
        choices = ' '.join(choices)
        question, choices, question_extension = template.format(original_question, choices,
                                                                question_extension).split('|')

    elif variation_type == 3:
        question = paraphrase
        n_chars_to_alter = 7
        question_extension = alter_text(question_extension, n_chars_to_alter)

        if dataset_type == 'mmvp':
            choices = original_choices.split('(b)')
            choices[1] = '(b)' + choices[1]
            choices = [part.strip() for part in choices]
        elif dataset_type == 'cvbench':
            choices = original_choices


        for i, part in enumerate(choices):
            try:
                choices[i] = part[:4] + alter_text(part[4:], 1)
            except:
                continue
        choices = ' '.join(choices)

    return question, choices, question_extension

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--n_variations_per_type", type=int, default=10)
    parser.add_argument("--prompt_templates_file", type=str)
    parser.add_argument("--stop_at", type=int, default=29)
    parser.add_argument("--start_at", type=int, default=0)
    args = parser.parse_args()

    para_f = open(os.path.join(args.root, 'Questions_Paraphrased.json'), 'r')
    paraphrases = json.load(para_f)

    templates_f = open(args.prompt_templates_file, 'r')
    templates = pd.read_csv(templates_f)

    variation_type = 0

    qs_ext = "Answer with the option's letter from the given choices directly."

    for var_i in range(args.n_variations_per_type*3):
        if var_i % args.n_variations_per_type == 0:
            variation_type += 1
        if var_i > args.stop_at:
            break
        if var_i < args.start_at:
            continue

        if 'MMVP' in args.root:
            benchmark_dir = os.path.join(args.root, 'Questions.csv')
            df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
            iterator_ = df.iterrows()
            qs_key, c_key = "Question", "Options"
            add_key = None
            dataset_type = 'mmvp'
        elif 'CV-Bench' in args.root:
            cvbench_dataset = load_dataset("parquet", data_files={"validation": os.path.join(args.root, "test.parquet")})
            iterator_ = enumerate(cvbench_dataset['validation'])
            qs_key, c_key = "question", "choices"
            add_key = "prompt"
            dataset_type = 'cvbench'

        if variation_type == 1:
            np.random.seed(2024+var_i)
            random.seed(2024+var_i)

        if dataset_type == 'mmvp':
            var_df = df.copy()
            df['question_extension'] = None
            var_df['question_extension'] = None
        else:
            var_dictarr = []

        for index, row in tqdm(iterator_):
            original_qs = row[qs_key]

            if dataset_type == 'cvbench':
                row[c_key] = ['('+chr(ord('A')+idx)+') '+ch for idx, ch in enumerate(row[c_key])]
                original_cs = '\n'.join(row[c_key])
            else:
                original_cs = row[c_key]

            row[qs_key], row[c_key], row['question_extension'] = generate_variation(variation_type,
                                                               paraphrases[index]['Paraphrased Questions'][var_i%args.n_variations_per_type],
                                                               templates['Template'][index%args.n_variations_per_type],
                                                               row[qs_key], row[c_key], qs_ext, dataset_type)
            if add_key is not None:
                row[add_key] = row[add_key].replace(original_qs, row[qs_key]).replace(original_cs, row[c_key])
                row[add_key] += '. ' + row['question_extension']

            if dataset_type == 'mmvp':
                var_df.iloc[index] = row
            else:
                var_dictarr.append(row)

        if dataset_type == 'mmvp':
            var_df = var_df.set_index('Index')
            var_df.to_csv(os.path.join(args.root, 'Questions_%02d.csv'%var_i))
        else:
            # save parquet
            var_df = pd.DataFrame(var_dictarr)
            var_df = var_df.drop('image', axis=1)
            var_df.to_parquet(os.path.join(args.root, "test_%02d.parquet"%var_i), index=False)
