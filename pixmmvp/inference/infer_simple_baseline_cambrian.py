import argparse
import json
import os
import shortuuid
import pandas as pd
from tqdm import tqdm
import cv2

import numpy as np
import torch
import random

import torch.nn.functional as F
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import torch.backends.cudnn as cudnn
from PIL import Image
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-path', type=str, default='')
    parser.add_argument('--batch-mode', action='store_true')
    parser.add_argument('--conv-mode', type=str, default='llava_v1')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--reg-tokens', type=int, default=0)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--root", type=str)
    parser.add_argument("--answers_file", type=str, default="answers.jsonl")
    parser.add_argument("--preds_dir", type=str, default="preds/")
    parser.add_argument("--viz_dir", type=str, default="viz/")
    parser.add_argument("--image_prefix", default="MMVP Images", type=str)
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.preds_dir):
        os.mkdir(args.preds_dir)

    if not os.path.exists(args.viz_dir):
        os.mkdir(args.viz_dir)

    # load models
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to(dtype=torch.float16)

    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    benchmark_dir = os.path.join(args.root, 'Questions.csv')
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")
    df_objects = pd.read_csv(os.path.join(args.root, 'Objects.csv'))

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Construct the 'prompts' string

        _, Object = df_objects.iloc[index]

        qs = f"Output bounding box for the {Object}."

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates['llama_3'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        photo_id = index+1
        image_path = os.path.join(args.root, args.image_prefix, f"{photo_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        image_width = image.width
        image_height = image.height
        image_tensor = process_images([image], image_processor, model.config)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True)

            # get answer
            input_token_len = input_ids.shape[1]
            answer = tokenizer.batch_decode(output_ids['sequences'], skip_special_tokens=True)[0]

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": photo_id,
                                       "prompt": qs,
                                       "answer": row["Correct Answer"],
                                       "response": answer,
                                       "answer_id": ans_id,
                                       "model_id": "Cambrian",
                                       }) + "\n")

            ans_file.flush()

            if '\n' in answer.strip():
                tokens = answer.split('\n')
            else:
                tokens = [answer.strip()]

            masks = []
            for token in tokens:
                bbox = [float(a) for a in token.strip()[1:-1].split(', ')]
                bbox[0] = int(bbox[0] * image_width)
                bbox[2] = int(bbox[2] * image_width)
                bbox[1] = int(bbox[1] * image_height)
                bbox[3] = int(bbox[3] * image_height)
                np_bbox = np.array(bbox)[None, :]

                sam_predictor.set_image(np.array(image))
                pred_mask, _, _ = sam_predictor.predict(box=np_bbox)
                pred_mask = pred_mask[-1]
                masks.append(pred_mask)

            final_mask = np.zeros(masks[0].shape).astype(bool)
            for mask in masks:
                final_mask[mask] = True

        image_vis = np.array(image).copy()
        image_vis[final_mask] = (0,0,255)
        image_vis = cv2.rectangle(image_vis, bbox[:2], bbox[2:], (0,0,255), 3)
        image_vis = Image.fromarray(image_vis)
        image_vis.save(os.path.join(args.viz_dir, f"%05d.jpg"%photo_id))
        cv2.imwrite(os.path.join(args.preds_dir, f"%05d.png"%photo_id), np.array(final_mask*255, np.uint8))

    ans_file.close()


