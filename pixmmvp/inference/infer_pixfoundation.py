import argparse
import json
import os
import shortuuid
import pandas as pd
from tqdm import tqdm
import cv2

import numpy as np
import torch
import spacy
import random

import torch.nn.functional as F
#import transformers
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

import torch.backends.cudnn as cudnn
from PIL import Image
from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor

from utils import group_tokens, merge_preds, get_spacy_embedding
import difflib

# COCO categories, used to filter out abstract noun phrases
seed_categories = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                   'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
                   'woman', 'man']

def get_overlap(s1, s2):
    seq = difflib.SequenceMatcher(a=s1.lower(),b=s2.lower())
    return seq.ratio()

def get_highest_overlap(groups, Object):
    highest_group = None
    highest_ratio = 0
    highest_idx = -1

#    max_spacy_score = 0
#    max_spacy_group = None
#    max_spacy_group_idx= -1

    for idx, group in enumerate(groups):
        ratio = get_overlap(group['phrase'], Object)
        if ratio > highest_ratio:
            highest_ratio = ratio
            highest_group = group
            highest_idx = idx

#        if group['spacy_score'] > max_spacy_score:
#            max_spacy_score= group['spacy_score']
#            max_spacy_group = group
#            max_spacy_group_idx = idx
#    return max_spacy_group, max_spacy_group_idx
    return highest_group, highest_idx

def process(line, args):
    qs = line["question"] + " Options:"
    options = line["options"].split('(b)')
    parts = [part.strip() for part in options]
    parts = [part.replace('(a)', 'A.').replace('(b)', 'B.') for part in parts]
    if len(parts) > 1:
        # parts[1] = "(b) " + parts[1]
        parts[1] = "B. " + parts[1]
    for part in parts:
        qs += f"\n{part}"
    qs += f"\n{args.question_extension}"
    return qs

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument('--image-path', type=str, default='')
    parser.add_argument('--output-path', type=str, default='')
    parser.add_argument('--batch-mode', action='store_true')
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--question', type=str, default='Describe the image in detail.')
    parser.add_argument('--conv-mode', type=str, default='llava_v1')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--feature-height', type=int, default=24)
    parser.add_argument('--feature-width', type=int, default=24)
    parser.add_argument('--reg-tokens', type=int, default=0)
    parser.add_argument('--tokenizer', type=str, default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--sam-model', type=str, default='vit_h')
    parser.add_argument('--sam-ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--offset', type=int, default=1)
    parser.add_argument('--aspect-ratio', type=str, default='pad')
    parser.add_argument('--spacy_score_thresh', type=float, default=0.70)
    parser.add_argument('--sam_score_thresh', type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--root", type=str)
    parser.add_argument("--prompt_for_seg", type=int, default=0)
    parser.add_argument("--answers_file", type=str, default="answers.jsonl")
    parser.add_argument("--preds_dir", type=str, default="preds/")
    parser.add_argument("--viz_dir", type=str, default="viz/")
    parser.add_argument("--meta_file", type=str, default="meta_file.txt")
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #transformers.enable_full_determinism(args.seed)

    if not os.path.exists(args.preds_dir):
        os.makedirs(args.preds_dir)

    if not os.path.exists(args.viz_dir):
        os.makedirs(args.viz_dir)

    # load models
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
#    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model.get_vision_tower().to(dtype=torch.float16)

    #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    spacy_model = spacy.load('en_core_web_lg')
    sam_model = sam_model_registry[args.sam_model](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    category_embeddings = [get_spacy_embedding(name, spacy_model) for name in seed_categories]
    category_embeddings = torch.stack(category_embeddings)

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

    f = open(args.meta_file, "w")

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Construct the 'prompts' string

        _, Object = df_objects.iloc[index]
        Object_embedding = get_spacy_embedding(Object, spacy_model)

        if args.prompt_for_seg == 1:
            qs = f"Identify {Object} in the scene."
        elif args.prompt_for_seg == 0:
            qs = row['Question'] + " " + row['Options']
        elif args.prompt_for_seg == 3:
            line = {
                    "question": str(row[1]),
                    "imageId": int(row[0])-1,
                    "options": str(row[2]),
                    "text_options": give_options(str(row[2])),
                    "answer": str(row[3])
                   }

            qs = process(line, args)

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        photo_id = index+1
        image_path = os.path.join(args.root, 'MMVP Images', f"{photo_id}.jpg")

        image = Image.open(image_path).convert('RGB')
        image_width = image.width
        image_height = image.height
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                #image_sizes=[image.size],
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
            answer = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]

            if args.prompt_for_seg == 3:
                # CVBench Format
                ans_file.write(json.dumps({"question_id": photo_id,
                                           "prompt": qs,
                                           "answer": answer.strip(),
                                           "gt_answer": row["Correct Answer"],
                                           "model_id": "LLava",
                                           "text_options": line["text_options"]
                                           }) + "\n")

            else:
                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({"question_id": photo_id,
                                           "prompt": qs,
                                           "answer": row["Correct Answer"],
                                           "response": answer,
                                           "answer_id": ans_id,
                                           "model_id": "LLava",
                                           }) + "\n")
            ans_file.flush()

            if args.prompt_for_seg == 3:
                continue

            # automatically detect image tokens
            image_token_start_index = -1
            for i in range(input_ids.shape[1]):
                if input_ids[0, i] == IMAGE_TOKEN_INDEX:
                    image_token_start_index = i
                    break
            assert image_token_start_index >= 0

            # process and save attention
            save_sequences = output_ids['sequences'][:, input_token_len:].detach().cpu()
            #save_sequences = output_ids['sequences'].detach().cpu()
            save_attn = []
            for i in range(len(output_ids['attentions'])):
                save_attn_i = output_ids['attentions'][i]                                           # n_layers x n_heads x n_output x n_input
                #save_attn_i = torch.cat([x[:, -2:-1, :] for x in save_attn_i])                      # n_layers x n_heads x 1 x n_input
                if i == 0:
                    save_attn_i = torch.cat([x[:, :, -2:-1, :] for x in save_attn_i])                      # n_layers x n_heads x 1 x n_input
                else:
                    save_attn_i = torch.cat(save_attn_i)
                if i == 0:
                    image_token_length = save_attn_i.shape[-1] - input_ids.shape[1] + 1
                    if args.reg_tokens > 0:
                        image_token_length -= args.reg_tokens
                        image_token_start_index += args.reg_tokens
                    image_token_end_index = image_token_start_index + image_token_length
                    assert image_token_length == args.feature_height * args.feature_width, \
                        f'Image token length mismatch: Expected {args.feature_height * args.feature_width}, got {image_token_length}'
                save_attn_i = save_attn_i[:, :, -1, image_token_start_index:image_token_end_index]  # n_layers x n_heads x n_image_tokens
                save_attn_i = save_attn_i.mean(dim=(0, 1))                                          # n_image_tokens
                save_attn_i = save_attn_i.reshape(args.feature_height, args.feature_width)          # feature_height x feature_width
                save_attn.append(save_attn_i.detach().cpu())

            save_attn = torch.stack(save_attn)

            attentions = save_attn.float()
            attn_mean = attentions.mean(dim=0)
            attentions = attentions - attn_mean

            # group tokens
            groups = group_tokens(save_sequences[0], tokenizer, spacy_model)
            if len(groups) == 0:
                cv2.imwrite(os.path.join(args.preds_dir, f"{photo_id}.png"), np.zeros((image_height, image_width), np.uint8))
                continue

            for group in groups:
                attns = attentions[group['tokens']]
                attn = attns.mean(dim=0)
                group['attention'] = attn

            # create segmentation masks
            group_scores = [group['attention'] for group in groups]
            group_scores = torch.stack(group_scores)

            if args.aspect_ratio == 'pad':
                upsample_size = max(image_height, image_width)
                crop_h_start = (upsample_size - image_height) // 2
                crop_h_end = crop_h_start + image_height
                crop_w_start = (upsample_size - image_width) // 2
                crop_w_end = crop_w_start + image_width
                upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                                size=(upsample_size, upsample_size),
                                                                mode='bicubic', align_corners=False).squeeze(0)
                upsample_scores = upsample_scores[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
            elif args.aspect_ratio == 'original':
                upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                                size=(image_height, image_width),
                                                                mode='bicubic', align_corners=False).squeeze(0)
            else:
                raise NotImplementedError(f'Invalid aspect ratio: {args.aspect_ratio}')

            sam_predictor.set_image(np.array(image))
            N, H, W = upsample_scores.shape
            max_indices = torch.argmax(upsample_scores.reshape(N, -1), dim=1)
            h_coords = max_indices // W
            w_coords = max_indices % W
            point_coords_np = torch.stack([w_coords, h_coords], dim=1).numpy()
            point_coords = torch.tensor(sam_predictor.transform.apply_coords(point_coords_np, sam_predictor.original_size)).unsqueeze(1).cuda()
            point_labels = torch.tensor([1] * N).unsqueeze(1).cuda()
            pred_masks, _, _ = sam_predictor.predict_torch(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
            assert pred_masks.shape[1] == 3, "Wrong number of masks from SAM"
            pred_masks = pred_masks.cpu().numpy()

            # filter based on spacy similarity and SAM score
            keep_groups = []
            for group_index in range(len(groups)):
                core_word = groups[group_index]['core_word']
                phrase = groups[group_index]['phrase']
                phrase_embedding = get_spacy_embedding(phrase, spacy_model)
                similarity = torch.cosine_similarity(phrase_embedding.unsqueeze(0), Object_embedding.unsqueeze(0), dim=1)
                mask = pred_masks[group_index]
                groups[group_index]['mask'] = mask
                keep_groups.append(group_index)
                color = np.random.randint(0, 256, 3)
                groups[group_index]['color'] = color
                area = mask.sum()
                groups[group_index]['area'] = area
                groups[group_index]['spacy_score'] = similarity

            if len(keep_groups) == 0:
                cv2.imwrite(os.path.join(args.preds_dir, f"{photo_id}.png"), np.zeros((image_height, image_width), np.uint8))
                continue

            groups = [groups[i] for i in keep_groups]
            pred_masks = pred_masks[keep_groups]

            #group, group_idx = get_highest_overlap(groups, Object)

            if group is None:
                cv2.imwrite(os.path.join(args.preds_dir, f"{photo_id}.png"), np.zeros((image_height, image_width), np.uint8))
                continue

            print(group['phrase'], '  : ', Object)

            for gridx, group in enumerate(groups):
                masks = group['mask']
                f.write("%d.jpg - %d - %s| %s| %f\n"%(photo_id, gridx, group['phrase'], group['core_word'], group["spacy_score"]))

                for midx, mask in enumerate(masks):
                    image_vis = np.array(image).copy()
                    image_vis[mask] = (0,0,255)
                    image_vis = cv2.circle(image_vis, point_coords_np[gridx], 5, (0,0,255), -1)
                    image_vis = Image.fromarray(image_vis)
                    image_vis.save(os.path.join(args.viz_dir, f"{photo_id}_%03d_%03d.jpg"%(gridx, midx) ))
                    cv2.imwrite(os.path.join(args.preds_dir, f"{photo_id}_%03d_%03d.png"%(gridx, midx)), np.array(mask*255, np.uint8))
            f.flush()

    ans_file.close()
    f.close()


