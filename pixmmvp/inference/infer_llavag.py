import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import random
import numpy as np
import torch.backends.cudnn as cudnn
import cv2

import transformers
from llava.eval.LLaVA_G_Eval import Evaluator_MM_Inter
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN

from PIL import Image
import math


import pandas as pd
from PIL import Image
import os

def preprocess_multi_conv(
    sources,
    tokenizer,
    has_image = False
):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conv.messages = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
    conv_prompt = conv.get_prompt()
    conv_prompt = "ASSISTANT: ".join(conv_prompt.split("ASSISTANT: ")[:-1]) + "ASSISTANT:"
    conv_prompt = conv_prompt.replace("</s>", "")
    conversations = [conv_prompt]
    print("Input Prompt: ", conv_prompt)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def filter_empty_box_mask(text, boxes_image, masks_image):
    def extract_text(sentence):
        # Use regular expression to find and extract the text and number
        import re
        pattern = r"<g_s>|<g_e> <seg>"
        cleaned_text = re.sub(pattern, '', sentence)
        return cleaned_text
    if len(boxes_image) == 0:
        return text, boxes_image, masks_image
    else:
        sub_texts = text.split(" <seg>")
        sub_texts_filtered = []
        boxes_image_filtered = []
        masks_image_filtered = []
        for box_per_gd, mask_per_gd, text_per_gd in zip(boxes_image, masks_image, sub_texts):
            text_per_gd += " <seg>"
            ind_nonempty_box = torch.where(box_per_gd.abs().sum(dim=1)>0)
            if len(ind_nonempty_box[0]) < box_per_gd.shape[0]:  # empty box encountered
                if len(ind_nonempty_box[0]) == 0:
                    text_per_gd = " " + " ".join(extract_text(text_per_gd).split())
                    sub_texts_filtered.append(text_per_gd)  # box is desperated
                    continue
                else:
                    box_per_gd = box_per_gd[ind_nonempty_box]
                    mask_per_gd = mask_per_gd[ind_nonempty_box]
                    boxes_image_filtered.append(box_per_gd)
                    masks_image_filtered.append(mask_per_gd)
                    sub_texts_filtered.append(text_per_gd)
            else:
                boxes_image_filtered.append(box_per_gd)
                masks_image_filtered.append(mask_per_gd)
                sub_texts_filtered.append(text_per_gd)
        sub_texts_filtered.append(sub_texts[-1])
        text_filtered = "".join(sub_texts_filtered)
        return text_filtered, boxes_image_filtered, masks_image_filtered


class InferenceLLavaG(object):
    def __init__(self,
                 model_path,
                 path_vision_cfg,
                 path_inter_cfg,
    ) -> None:
        self.model_backend = Evaluator_MM_Inter(
            model_path=model_path,
            path_vision_model_cfg=path_vision_cfg,
            path_inter_model_cfg =path_inter_cfg,
        )
        self.model_backend.data_mapper.preprocess = preprocess_multi_conv

    def inference(self, data_dict):
        # TODO: Implement data_mapper.
        data_dict = self.model_backend.data_mapper(data_dict)[0]
        #
        device = self.model_backend.model.device
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)

        response_text, response_boxes, response_mask, mask_inter = self.model_backend.evaluate_sample([data_dict])
        #
        response_text, response_boxes, response_mask = filter_empty_box_mask(response_text, response_boxes, response_mask)
        return response_text, response_boxes, response_mask, mask_inter



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def post_process_text_response(text):
    def find_start_idxes(sentence, word):
        window_size = len(word)
        start_indexes = []
        assert len(sentence) > window_size
        if sentence == window_size:
            return [0]
        for start_index in range(len(sentence) - window_size):
            if sentence[start_index: start_index + window_size] == word:
                start_indexes.append(start_index)
        return start_indexes
    def add_color_to_text(obj_id, text):
        color = colors[obj_id]
        text = f"<span style='color: rgb{color};'>{text}</span>"
        return text
    def format_sentence(splitted_sentence):
        joint_sentence = " ".join(splitted_sentence)
        return joint_sentence
    def extract_text(sentence):
        import re
        pattern = r"<g_s>|<g_e>"
        cleaned_text = re.sub(pattern, '', sentence)
        return cleaned_text

    text_pure = ""
    seg_start_index = find_start_idxes(text, "<seg>")
    if len(seg_start_index) > 0:
        count_obj = 0
        subtexts = text.split(" <seg>")
        for subtext in subtexts:
            if "<g_s>" in subtext:
                start_idx = find_start_idxes(subtext, "<g_s>")[0]
                text_pure = format_sentence([text_pure, format_sentence(subtext[:start_idx].split())])
                text_ = extract_text(subtext[start_idx:])
                text_pure += add_color_to_text(count_obj, text_)
                count_obj += 1
            else:
                text_pure = format_sentence([text_pure, format_sentence(subtext.split())])
    else:
        text_pure = text
    return text_pure

def post_process_masks(path_ori_image, masks_gd, path_save_gd):
    def unresize_mask(mask, width, height):
        import torch.nn.functional as F
        if width >= height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            mask = F.interpolate((mask[None, ...]).float(), size=[width, width], mode="nearest")[0]
            mask = mask[:, :height]
        elif width < height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            mask = F.interpolate((mask[None, ...]).float(), size=[height, height], mode="nearest")[0]
            mask = mask[:, :, :width]
        return mask
    def unnormalize_inter(mask, loc):
        height, width, _ = mask.shape
        loc_x_mean, loc_y_mean, loc_w, loc_h = loc
        if height >= width:
            loc_x_mean = loc_x_mean / (width/height)
            loc_w = loc_w / (width/height)
        else:
            loc_y_mean = loc_y_mean / (height/width)
            loc_h = loc_h / (height/width)
        return [loc_x_mean-loc_w/2, loc_y_mean-loc_h/2, loc_x_mean+loc_w/2, loc_y_mean+loc_h/2]
    image = cv2.imread(path_ori_image)
    gd_image = image.copy()
    preds = None
    if not (masks_gd is None):
        height, width = image.shape[:2]
        masks_gd = [unresize_mask(aa, width, height) for aa in masks_gd]
        colored_mask = torch.zeros((3, height, width), dtype=torch.long)
        for gd_id, gd_mask_result in enumerate(masks_gd):
            gd_mask_result = gd_mask_result.sum(dim=0, keepdim=True)
            current_preds = (gd_mask_result[0] > 0.5).detach().cpu()
            if preds is None:
                preds = current_preds
            else:
                preds[current_preds==1] = 1
            colored_mask[:, gd_mask_result[0] > 0.5] = torch.tensor([0,0,255])[:, None]
            tmp_image = (gd_image * 0.5 + colored_mask.permute(1,2,0).numpy() * 0.5).astype(np.uint8)
            gd_image[current_preds] = tmp_image[current_preds]

        cv2.imwrite(path_save_gd, gd_image)
    return preds, image

def post_process_gd_response(path_ori_image, gd_results_per_image):
    def unresize_box(box, width, height):
        ratio = min(width, height) / max(width, height)
        if width > height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            box[:, 1] = box[:, 1] / ratio
            box[:, 3] = box[:, 3] / ratio
        elif width < height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            box[:, 0] = box[:, 0] / ratio
            box[:, 2] = box[:, 2] / ratio
        return box
    image = cv2.imread(path_ori_image)
    height, width = image.shape[:2]
    gd_results_per_image = [unresize_box(aa.detach().cpu(), width, height) for aa in gd_results_per_image]
    for gd_id, gd_result in enumerate(gd_results_per_image):
        bboxes = gd_result.cpu().tolist()
        for bbox in bboxes:
            bbox = [int(bbox[0]*width), int(bbox[1]*height), int(bbox[2]*width), int(bbox[3]*height)]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[gd_id][::-1], 2)
    path_save = get_image_name(prefix="grounding_img_")
    cv2.imwrite(path_save, image)
    return (path_save, )

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

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

def eval_model(args):
    our_chatbot = InferenceLLavaG(args.model_path, args.path_vision_cfg, args.path_inter_cfg)

    benchmark_dir = os.path.join(args.directory, 'Questions.csv')
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")
    df_objects = pd.read_csv(os.path.join(args.directory, 'Objects.csv'))

    if args.preds_dir != "":
        if not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)
        if not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Construct the 'prompts' string
        photo_id = index+1
        image_path = os.path.join(args.directory, 'MMVP Images', f"{photo_id}.jpg")

        cur_prompt = row['Question'] + " " + row['Options']

        _, Object = df_objects.iloc[index]
        if args.prompt_for_seg == 1:
#            cur_prompt = f'Where is {Object} in the image? (with grounding)'
            cur_prompt = f'Can you identify {Object} in this image? (with grounding)'
        elif args.prompt_for_seg == 2:
            cur_prompt += f' Where is {Object} in the image? (with grounding)'
        elif args.prompt_for_seg == 3:
            line = {
                    "question": str(row[1]),
                    "imageId": int(row[0])-1,
                    "options": str(row[2]),
                    "text_options": give_options(str(row[2])),
                    "answer": str(row[3])
                   }

            cur_prompt = process(line, args)

        input_data_dict = {'file_name': image_path, 'image_id': 0, 'question_id': 0,
                           'conversations': [[[{'from': 'human', 'value': '<image> '+cur_prompt},
                                               {'from': 'gpt', 'value': 'Placeholder.'}], None]]}
        input_data_dict["points"] = None
        input_data_dict["mode_inter"] = None
        input_data_dict["matching_threshold"] = 0.05
        input_data_dict["temporature"] = args.temperature

        response_text, _, response_mask, _  = our_chatbot.inference(input_data_dict)

        if args.preds_dir != "":
            response_msks, img  = post_process_masks(image_path, response_mask, os.path.join(args.viz_dir, f"{photo_id}.jpg"))
            if response_msks is None:
                response_msks = np.zeros(img.shape[:2], np.uint8)
            else:
                response_msks = np.array(response_msks.detach().cpu()*255, np.uint8)

            cv2.imwrite(os.path.join(args.preds_dir, "%05d.png"%photo_id), response_msks)

        print(response_text)
        if args.prompt_for_seg == 3:
            # CVBench Format
            ans_file.write(json.dumps({"question_id": photo_id,
                       "prompt": cur_prompt,
                       "answer": response_text.strip(),
                       "gt_answer": row["Correct Answer"],
                       "model_id": "LLava-G",
                       "text_options": line["text_options"]
                       }) + "\n")

        else:
            # GPT Grader Format
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": photo_id,
                                       "prompt": cur_prompt,
                                       "answer": row["Correct Answer"],
                                       "response": response_text,
                                       "answer_id": ans_id,
                                       "model_id": "LLava-G",
                                       }) + "\n")
        ans_file.flush()
    ans_file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--directory", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--path_vision_cfg", default="", type=str)
    parser.add_argument("--path_inter_cfg", default="", type=str)
    parser.add_argument("--prompt_for_seg", default="0", type=int)
    parser.add_argument("--preds_dir", default="", type=str)
    parser.add_argument("--viz_dir", default="", type=str)
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")

    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    #transformers.enable_full_determinism(args.seed)

    eval_model(args)
