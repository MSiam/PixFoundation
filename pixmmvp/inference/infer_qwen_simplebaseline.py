import torch
import torchvision
from PIL import Image
import numpy as np
import os
import cv2
import argparse
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import itertools

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from segment_anything import sam_model_registry, SamPredictor

import json
import markdown
from bs4 import BeautifulSoup

#################################################### Helper Functions ##############################
def parse_json(response):
    # Meta response for now is not being used, is used in InternVL Variant not Qwen
    html = markdown.markdown(response, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    json_text = soup.find('code').text
    data = json.loads(json_text)
    return data

def parse_response(response, meta_response, response_type='point'):
    input_height, input_width, (width, height) = meta_response
    try:
        json_output = parse_json(response)
    except:
        out = None

    if response_type == 'point':
        try:
            out = []
            for point in json_output:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(point["point_2d"][1]/input_height * height)
                abs_x1 = int(point["point_2d"][0]/input_width * width)
                out.append([abs_x1, abs_y1])
        except:
            # Occurs when the returned text has no json format bounding boxes at all
            out = None
    elif response_type == 'box':
        try:
            out = []
            for bounding_box in json_output:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
                abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
                abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
                abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

                if abs_x1 > abs_x2:
                  abs_x1, abs_x2 = abs_x2, abs_x1

                if abs_y1 > abs_y2:
                  abs_y1, abs_y2 = abs_y2, abs_y1
                out.append([abs_x1, abs_y1, abs_x2, abs_y2])
        except:
            # Occurs when the returned text has no json format bounding boxes at all
            out = None
    return out

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def process(line, question_extension):
    qs = line["question"] + " Options:"
    options = line["options"].split('(b)')
    parts = [part.strip() for part in options]
    parts = [part.replace('(a)', 'A.').replace('(b)', 'B.') for part in parts]
    if len(parts) > 1:
        # parts[1] = "(b) " + parts[1]
        parts[1] = "B. " + parts[1]
    for part in parts:
        qs += f"\n{part}"
    qs += f"\n{question_extension}"
    return qs

def retrieve_permutations(pred_masks):
    if pred_masks.shape[0] == 1 and pred_masks.shape[1] == 1:
        return pred_masks[0]

    MAX_INST = 10
    MED_INST = 5

    if pred_masks.shape[0] > MAX_INST:
        # Use only one mask per instance
        pred_masks = pred_masks[:, :1]
    elif pred_masks.shape[0] > MED_INST:
        # Use two masks per instance generated from SAM accomodating ambiguity
        pred_masks = pred_masks[:, :2]

    final_masks = []
    list_ids = [['%d_%d'%(i, j) for j in range(pred_masks.shape[1])] for i in range(pred_masks.shape[0])]
    all_pairs = itertools.product(*list_ids)

    for pair in all_pairs:
        curr_mask = torch.zeros(pred_masks.shape[-2:]).cuda()
        for element in pair:
            inst_id, mask_id = element.split('_')
            curr_mask += pred_masks[int(inst_id)][int(mask_id)]
        curr_mask[curr_mask > 0] = 1
        final_masks.append(curr_mask)

    final_masks = torch.stack(final_masks, axis=0)
    return final_masks

#################################################### Main Functions ##############################
def image_inference(img_url, prompt, model, processor, system_prompt="You are a helpful assistant",
                    max_new_tokens=1024, temperature=0, resize=False):

    image = Image.open(img_url)
    if resize:
        image = image.resize((image.size[0]*2, image.size[1]*2))
    messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=1024)#, temperature=temperature)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14
    original_size = image.size
    if resize:
        original_size = (image.size[0]//2, image.size[1]//2)
    return output_text[0], (input_height, input_width, original_size)

def vis_boxes(boxes, img_path):
    img = np.array(Image.open(img_path))

    for box in boxes:
        start = box[:2]
        end = box[2:]
        img = cv2.rectangle(img, start, end, (255, 0, 0), 2)

    plt.imshow(img);plt.show()

def vis_points(points, img_path):
    img = np.array(Image.open(img_path))

    for point in points:
        img = cv2.circle(img, point, 10, (255, 0, 0), 2)

    plt.imshow(img);plt.show()

def merge_masks(pred_masks):
    final_mask = np.zeros(pred_masks.shape[-2:], np.uint8)
    for mask in pred_masks:
        final_mask[mask>0] = 1
    return final_mask

def check_boundaries(boxes, size):
    for idx in range(len(boxes)):
        if boxes[idx, 0] < 0:
            boxes[idx, 0] = 0
        if boxes[idx, 2] >= size[0]:
            boxes[idx, 2] = size[0] - 1

        if boxes[idx, 1] < 0:
            boxes[idx, 1] = 0
        if boxes[idx, 3] >= size[1]:
            boxes[idx, 3] = size[1] - 1

    return boxes

def inference(args):

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    token_id = 0

    # Load Model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, device_map='cuda', torch_dtype=torch.bfloat16,
                                                               attn_implementation="flash_attention_2")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    # Load SAM model
    sam_model = sam_model_registry['vit_h'](checkpoint=args.sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)

    # Prepare Dataloader
    benchmark_dir = os.path.join(args.directory, args.questions_file)
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")
    df_objects = pd.read_csv(os.path.join(args.directory, args.objects_file))

    if args.preds_dir != "":
        if not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)
        if not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)

    question_extension = args.question_extension

    if args.grounding_prompts_file != '':
        grounding_prompts = pd.read_csv(os.path.join(args.directory, args.grounding_prompts_file),
                                        index_col=0)

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        if index < args.start_index:
            continue

        # Construct the 'prompts' string
        photo_id = index+1
        image_path = os.path.join(args.directory, args.image_prefix, f"{photo_id}.jpg")
        image = Image.open(image_path)

        cur_prompt = row['Question'] + " " + row['Options']

        _, Object = df_objects.iloc[index]
        if args.prompt_for_seg == 1:
            if args.prompt_type == 'point':
                cur_prompt = f'Point to the {Object}, output its coordinates in point-format as JSON'
            elif args.prompt_type == 'box':
                #cur_prompt = f'Locate the {Object} and output all the coordinates in JSON format.'
                if args.variation_idx != -1 and grounding_prompts is not None:
                    cur_prompt = grounding_prompts['Template'].iloc[args.variation_idx].format(Object)
                else:
                    cur_prompt = f'Locate the {Object} and output all the coordinates in JSON format for a tight bounding box surrounding the object. If the object does not exist in the image dont generate any boxes.'
        elif args.prompt_for_seg == 3:
            line = {
                    "question": str(row[1]),
                    "imageId": int(row[0])-1,
                    "options": str(row[2]),
                    "text_options": give_options(str(row[2])),
                    "answer": str(row[3])
                   }

            if 'question_extension' in row:
                question_extension = row['question_extension']

            cur_prompt = process(line, question_extension)

        ######## Infer Qwen2.5-VL + SAM
        response, meta_response = image_inference(image_path, cur_prompt, model, processor, temperature=args.temperature,
                                                  resize=(args.prompt_type=='box'))
        if args.prompt_for_seg == 1:
            prompt_coords = parse_response(response, meta_response, response_type=args.prompt_type)

            sam_predictor.set_image(np.array(image))
            if prompt_coords is not None and len(prompt_coords) != 0:
                prompt_coords_np = np.array(prompt_coords)

                if args.prompt_type == 'point':
                    prompt_coords = torch.tensor(sam_predictor.transform.apply_coords(prompt_coords_np,
                                                 sam_predictor.original_size)).unsqueeze(1).cuda()
                    point_labels = torch.tensor([1] * prompt_coords.shape[0]).unsqueeze(1).cuda()
                    pred_masks, _, _ = sam_predictor.predict_torch(point_coords=prompt_coords, point_labels=point_labels, multimask_output=True)
                    assert pred_masks.shape[1] == 3, "Wrong number of masks from SAM"

                elif args.prompt_type == 'box':
                    prompt_coords_np = check_boundaries(prompt_coords_np, image.size)
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(torch.tensor(prompt_coords_np), sam_predictor.original_size)
                    pred_masks, _, _ = sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                                   boxes=transformed_boxes.cuda(), multimask_output=False)
                pred_masks = np.asarray(pred_masks[:, -1].cpu(), np.uint8) # always retrieve the last
                pred_masks = merge_masks(pred_masks)
            else:
                pred_masks = np.zeros((image.size[1], image.size[0]), np.uint8)

            if args.preds_dir != "":
                cv2.imwrite(os.path.join(args.preds_dir, "%05d.png"%photo_id),
                            np.asarray(pred_masks, np.uint8)*255)

        ######## Process predicted masks for saving
        torch.cuda.empty_cache()

        if args.prompt_for_seg == 3:
            # CVBench Format
            ans_file.write(json.dumps({"question_id": photo_id,
                       "prompt": cur_prompt,
                       "answer": response.strip(),
                       "gt_answer": row["Correct Answer"],
                       "model_id": "Qwen2.5-VL",
                       "text_options": line["text_options"]
                       }) + "\n")

        else:
            # GPT Grader Format
            ans_file.write(json.dumps({"question_id": photo_id,
                                       "prompt": cur_prompt,
                                       "answer": row["Correct Answer"],
                                       "response": response,
                                       "answer_id": photo_id,
                                       "model_id": "Qwen2.5-VL",
                                       }) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--directory", type=str, default="")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--prompt_for_seg", default="0", type=int)
    parser.add_argument("--preds_dir", default="", type=str)
    parser.add_argument("--viz_dir", default="", type=str)
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--questions_file", default="Questions.csv", type=str)
    parser.add_argument("--image_prefix", default="MMVP Images", type=str)
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--prompt_type', type=str, default='point')
    parser.add_argument("--grounding_prompts_file", default="", type=str)
    parser.add_argument("--variation_idx", default=-1, type=int)
    parser.add_argument("--objects_file", default="Objects.csv", type=str)
    args = parser.parse_args()

#    cfg = setup_cfg(args)
    inference(args)

