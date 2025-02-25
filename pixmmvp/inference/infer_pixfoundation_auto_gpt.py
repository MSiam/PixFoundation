import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
import csv
from PIL import Image
import itertools
import glob
import pandas as pd
import cv2
import re
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import openai

import time
from PIL import Image
import math


def visualize_preds(preds_files, image):
    vis_images = []
    color = (255, 0, 0)

    for pred_file in preds_files:
        temp_image = np.array(image).copy()
        mask = cv2.imread(pred_file, 0)
        mask[mask==255] = 1
        _mask_image = np.zeros((temp_image.shape[0], temp_image.shape[1], 3), dtype=np.uint8)
        if mask.sum() != 0:

            _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
            _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
            _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]
        else:
            mask = np.zeros((temp_image.shape[0], temp_image.shape[1]), dtype=np.uint8)

        overlay_img = temp_image * 0.5 + _mask_image * 0.5
        temp_image[mask==1] = overlay_img[mask==1]
        vis_images.append(Image.fromarray(temp_image))

    return vis_images

def process_firststage(line, args, images, preds_files):
    image_id = line["imageId"]
    filename= str(image_id+1)
    matched_preds_files = [pf for pf in preds_files if filename == pf.split('/')[-1].split('.')[0].split('_')[0]]
    matched_preds_files = sorted(matched_preds_files)
    input_image = images[image_id]
    vis_images = visualize_preds(matched_preds_files, input_image)
    return matched_preds_files, vis_images, filename

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

NUM_SECONDS_TO_SLEEP = 10
def prompt_gpt(prompt, urls):
    img_urls_dicts = []

    for url in urls:
        img_urls_dicts.append({
            "type": "image_url",
            "image_url": {
                "url": url,},
        })

    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                    }, {
                    'role': 'user',
                    'content':
                    [{
                        "type": "text",
                        "text": prompt,
                    }] + img_urls_dicts,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    return answer



def div_images_into_groups(args, base_url, Object, matched_preds_files):

    ngroupimgs = len(matched_preds_files)//3
    best_indices = []
    best_files =[]

    for grimg_idx in range(ngroupimgs):
        if grimg_idx == ngroupimgs - 1:
            img_urls = []
            for img_file in matched_preds_files[grimg_idx*3:]:
                img_urls.append(base_url + img_file)
        else:
            img_urls = []
            for img_file in matched_preds_files[grimg_idx*3:(grimg_idx+1)*3]:
                img_urls.append(base_url + img_file)

        N = len(img_urls)
        prompt = f"Select the image that has {Object} best highlighted in red color than the others? Answer with a number from 1 to {N}. Mention the number only."
        outputs = prompt_gpt(prompt, img_urls)

        try:
            # If output is directly a number
            best_indices.append(grimg_idx*3+int(outputs)-1)
            best_files.append(matched_preds_files[best_indices[-1]])
        except:
            # If output is not directly a number we need a regex matching
            try:
                best_indices.append(int(re.search("\d+", outputs)[0])-1)
                best_files.append(matched_preds_files[best_indices[-1]])
            except:
                # If output doesn't show a number like third/first/...or somethign quite different can not be matched directly into a number
                # Use the first image as default
                best_indices.append(0)
                best_files.append(matched_preds_files[best_indices[-1]])
    return best_indices, best_files

def eval_model(args):
    # Model
    images = {}
    for i in range(300):
        file_path = os.path.join(args.root, 'MMVP Images', '%d.jpg'%(i+1))
        images[i] = Image.open(file_path).convert('RGB')


    questions = []
    file_path = os.path.join(args.root, 'Questions.csv')
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0]=="lndex":
                continue
            if row[0]=="Index":
                continue
            questions.append({
                "question": str(row[1]),
                "imageId": int(row[0])-1,
                "options": str(row[2]),
                "text_options": give_options(str(row[2])),
                "answer": str(row[3])
            })

    answers_file = os.path.expanduser(args.answers_file)
    if not answers_file.endswith(".jsonl"):
        raise ValueError("Answers file must be a jsonl file")

    if not os.path.exists(args.auto_vis_dir):
        os.mkdir(args.auto_vis_dir)

    preds_files = sorted(glob.glob(os.path.join(args.preds_dir, '*.png')))
    df_objects = pd.read_csv(os.path.join(args.root, 'Objects.csv'))
    ans_file = open(answers_file, "w")

    idx = -1
    final_selections = {}

    model_dir = args.auto_vis_dir.split('/')[-2]
    base_github_url = "https://raw.githubusercontent.com/MSiam/AutoGPTImages/eb98403d6763f16505ffe45fe0a746c3c4365aab/"
    base_github_url = base_github_url + model_dir + '/'
    print('Base url: ', base_github_url)

    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        _, Object = df_objects.iloc[idx]

        if args.stage == 1:
            matched_preds_files, vis_images, filename = process_firststage(line, args, images, preds_files)
            filedir = os.path.join(args.auto_vis_dir, filename)

            if not os.path.exists(filedir):
                os.mkdir(filedir)

            for vimg, mpred in zip(vis_images, matched_preds_files):
                vimg.save(os.path.join(filedir, mpred.split('/')[-1]))

        elif args.stage == 2:
            image_id = line["imageId"]
            filename= str(image_id+1)
            matched_preds_files = os.listdir(os.path.join(args.auto_vis_dir, filename))

            if len(matched_preds_files) > 1:
                img_urls = []
                for mpfile in matched_preds_files:
                    img_urls.append(base_github_url + filename + '/' + mpfile)

                N = len(img_urls)
                prompt = f"Select the image that has {Object} best highlighted in red color than the others? Answer with a number from 1 to {N}. Mention the number only."
                outputs = prompt_gpt(prompt, img_urls)

                try:
                    # If output is directly a number
                    best_idx = int(outputs)-1
                except:
                    # If output is not directly a number we need a regex matching
                    try:
                        best_idx = int(re.search("\d+", outputs)[0])-1
                    except:
                        # If output doesn't show a number like third/first/...or somethign quite different can not be matched directly into a number
                        # Use the first image as default
                        best_idx = 0

                selected_file = os.path.join(args.preds_dir, matched_preds_files[best_idx])

                ans_file.write(json.dumps({
                    "question_id": idx,
                    "selected file": selected_file,
                }) + "\n")
                ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--preds_dir", default="", type=str)
    parser.add_argument("--auto_vis_dir", default ="", type=str)
    parser.add_argument("--stage", default=1, type=int)
    parser.add_argument("--answers_file", type=str, default="./answers.jsonl")
    parser.add_argument('--openai_api_key', default = "", help='Your OpenAI API key')

    args = parser.parse_args()
    if args.openai_api_key != "":
        openai.api_key = args.openai_api_key

    eval_model(args)


