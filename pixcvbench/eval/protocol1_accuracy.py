from tqdm import tqdm
import argparse
import json
import openai
import re
import time
import numpy as np
import pandas as pd
from datasets import load_dataset
import os

# Create the parser
parser = argparse.ArgumentParser(description='Process OpenAI API key and JSONL file path.')

# Add arguments
parser.add_argument('--openai_api_key', default = "", help='Your OpenAI API key')
parser.add_argument('--answer_file', default = "answer.jsonl",help='Path to the JSONL file')
parser.add_argument('--questions_file', default='', type=str)

# Parse arguments
args = parser.parse_args()

openai.api_key = args.openai_api_key
NUM_SECONDS_TO_SLEEP = 10
# Define a function to query the OpenAI API and evaluate the answer
def get_yes_no_answer(question):
    while True:
        try:
            response = openai.ChatCompletion.create(
                #model='gpt-4-0314',
                model='gpt-4o',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no.'
                }, {
                    'role': 'user',
                    'content': question,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            #print(e)
            continue
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    #print(answer)
    yes_no_regex = re.compile(r"^(yes|no)\.*$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.lower()
    else:
        return "Could not determine yes or no."


def retrieve_full_acc(answer_file, questions_file):
    if questions_file != '':
        cvbench_dataset = load_dataset("parquet", data_files={"validation": questions_file})
        if 'ade' in answer_file:
            data_type = 'ADE20K'
        else:
            data_type = 'COCO'
        questions = [entry['prompt'] for entry in cvbench_dataset['validation'] if entry['source'] == data_type]
    else:
        cvbench_dataset = None


    accuracy = []
    num_correct, num_total = 0, 0
    with open(answer_file, 'r') as file:
        index, round_correct = 0, 0
        for line in tqdm(file):
            data = json.loads(line)
            try:
                # Protocol 1 format
                question, correct_answer, model_response = data["prompt"], data["answer"], data["response"]
            except:
                # Protocol 2 format for confirmation on the prompt sensitivity results
                assert cvbench_dataset is not None, "CVBench dataset cant be none in this evaluation"
                qid, correct_answer, model_response = data['questionId'], data["gt_answer"], data["answer"]
                if data_type == 'COCO':
                    qid -= 633 # COCO #805 after ADE #633

                question = questions[qid]
            question4gpt = f"Given the following question {question}, the correct answer is {correct_answer}. Does the following answer correctly answers the question, answer:{model_response}? Respond with a Yes/No"
            #print(question, ' ', correct_answer, ' ', model_response)
            gpt_grade = get_yes_no_answer(question4gpt)

            index += 1

            if gpt_grade=="yes" or gpt_grade=="yes.":
                accuracy.append(1)
                round_correct += 1
                #print('correct')
                #print('===========================================================================')
            else:
                accuracy.append(0)

            if index == 2:
                index = 0
                if round_correct == 2:
                    num_correct += 1
                round_correct = 0

                num_total += 1
    print(f"The accuracy is {num_correct/num_total}")
    return accuracy

acc = retrieve_full_acc(args.answer_file, args.questions_file)
dataframe = pd.DataFrame(acc, columns=['acc'])
dataframe.to_csv(args.answer_file.replace('.jsonl', '.csv'))
