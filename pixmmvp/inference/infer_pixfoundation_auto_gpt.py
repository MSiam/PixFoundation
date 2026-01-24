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
import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        base64_image = encode_image_to_base64(url)
        url = f"data:image/jpeg;base64,{base64_image}"
        img_urls_dicts.append({
            "type": "image_url",
            "image_url": {
                "url": url,},
        })

    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-5.1',
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
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation (if I am prompting it w/ same prompt multiple times)
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer = response['choices'][0]['message']['content']
    return answer

def eval_model(args):
    # Model
    images = {}
    for i in range(300):
        file_path = os.path.join(args.root, args.image_prefix, '%d.jpg'%(i+1))
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

    MAX_IMGS = 499
    for line in tqdm(questions, total=len(questions)):
        idx = idx+1
        _, Object = df_objects.iloc[idx]
        if idx < args.start_index:
            continue

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

            # Filter out images that do not have the object
            prompt = f"Does this image have {Object}? Answer yes/no only."
            file_path = os.path.join(args.root, args.image_prefix, filename+'.jpg')
            yes_no_answer = prompt_gpt(prompt, [file_path])
            yes_regex = re.compile(r"^(yes)\.*$", re.IGNORECASE)
            if not yes_regex.match(yes_no_answer.lower()):
                selected_file = 'NONE'
                ans_file.write(json.dumps({
                    "question_id": idx,
                    "selected file": selected_file,
                }) + "\n")
                ans_file.flush()
                continue

            # Perform Mask Selection
            matched_preds_files = os.listdir(os.path.join(args.auto_vis_dir, filename))

            if len(matched_preds_files) > 1:
                img_urls = []
                for mpfile in matched_preds_files:
                    img_urls.append(os.path.join(args.auto_vis_dir, filename, mpfile))

                if len(img_urls) > MAX_IMGS:
                    N = MAX_IMGS
                    img_urls = img_urls[:MAX_IMGS]

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

            elif len(matched_preds_files) == 1:
                # its only 1 file
                selected_file = os.path.join(args.preds_dir, matched_preds_files[0])

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
    parser.add_argument("--image_prefix", type=str, default="MMVP Images")
    parser.add_argument("--start_index", type=int, default=0)
    args = parser.parse_args()
    if args.openai_api_key != "":
        openai.api_key = args.openai_api_key

    eval_model(args)


