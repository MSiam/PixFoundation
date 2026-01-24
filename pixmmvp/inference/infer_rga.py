import argparse
import json
import os
import sys
from tqdm import tqdm
from glob import glob
import pandas as pd
import torch.backends.cudnn as cudnn
import random
import shortuuid
sys.path.append(".")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
from qwen_vl_utils import process_vision_info

# from model.segment_anything.utils.transforms import ResizeLongestSide
# from model.qwen_2_5_vl import UniGRConfig, UniGRModel
from utils.utils import DirectResize
from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import get_sparse_indices, dict_to_cuda, preprocess


def parse_args(args):
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--dataset_root")
    parser.add_argument("--version", default="PATH/TO/MODEL")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--prompt_for_seg", default="0", type=int)
    parser.add_argument("--preds_dir", default="", type=str)
    parser.add_argument("--viz_dir", default="", type=str)
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--num_frames_mllm", default=4, type=int)
    parser.add_argument("--max_pixels", default=384*28*28, type=int)
    parser.add_argument("--inference_mode", default="video")
    parser.add_argument("--postproc", default="simple")
    parser.add_argument("--questions_file", default="Questions.csv", type=str)
    parser.add_argument("--image_prefix", default="MMVP Images", type=str)
    parser.add_argument("--grounding_prompts_file", default="", type=str)
    parser.add_argument("--variation_idx", default=-1, type=int)
    parser.add_argument("--objects_file", default="Objects.csv", type=str)
    return parser.parse_args(args)

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

def main(args):
    # ---------------------------- config env ------------------------------------
    args = parse_args(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create model
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # ---------------------------- prepare model ------------------------------------
    model_args = {
        "train_mask_decoder": False,
        "seg_token_idx": args.seg_token_idx,
    }
    config = UniGRConfig.from_pretrained(
        args.version,
        **model_args,
    )
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    else:
        raise NotImplementedError

    transform = DirectResize(args.image_size)

    model.eval()

    # ---------------------------- read data ------------------------------------
    benchmark_dir = os.path.join(args.dataset_root, args.questions_file)
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Now open the file
    ans_file = open(answers_file, "w")
    df_objects = pd.read_csv(os.path.join(args.dataset_root, args.objects_file))

    if args.preds_dir != "":
        if not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)
        if not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)

    if args.grounding_prompts_file != '':
        grounding_prompts = pd.read_csv(os.path.join(args.dataset_root, args.grounding_prompts_file),
                                        index_col=0)

    # Loop through each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Construct the 'prompts' string
        photo_id = index+1
        image_path = os.path.join(args.dataset_root, args.image_prefix, f"{photo_id}.jpg")

        cur_prompt = row['Question'] + " " + row['Options']
        cur_prompt += " Answer the question. Please answer in one sentence."
        _, Object = df_objects.iloc[index]

        if args.prompt_for_seg == 1:
            # This prompt resulted in better results than the one used in Fig.1 of their paper
            if args.variation_idx != -1 and grounding_prompts is not None:
                cur_prompt = grounding_prompts['Template'].iloc[args.variation_idx].format(Object)
            else:
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
            if 'question_extension' in row:
                question_extension = row['question_extension']
            else:
                question_extension = args.question_extension

            cur_prompt = process(line, question_extension)

        if args.inference_mode == "video":
            image_file_list = [image_path] * args.num_frames_mllm
            total_frames = args.num_frames_mllm
            sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)

            # pre-process images
            frames_list, image_list_sam, image_list_np = [], [], []

            for frm_idx in sparse_idxs:
                image_path = image_file_list[frm_idx]
                image_pil = Image.open(image_path).convert("RGB")
                frames_list.append(image_pil)

            for frm_idx in range(total_frames):
                image_path = image_file_list[frm_idx]
                image_np = cv2.imread(image_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                original_size_list = [image_np.shape[:2]]

                image = transform.apply_image(image_np)
                resize_list = [image.shape[:2]]

                image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
                if args.precision == "bf16":
                    image = image.bfloat16()
                elif args.precision == "fp16":
                    image = image.half()
                else:
                    image = image.float()

                image_list_sam.append(image)
                image_list_np.append(image_np)

            # prepare text query and prompt
            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": frames_list, "max_pixels": args.max_pixels},
                    {"type": "text", "text": cur_prompt}
                ]}
            ]

        else:

            # pre-process images
            image_list_sam = []

            for frm_idx in range(total_frames):
                image_np = cv2.imread(image_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                original_size_list = [image_np.shape[:2]]

                image = transform.apply_image(image_np)
                resize_list = [image.shape[:2]]

                image = (preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
                if args.precision == "bf16":
                    image = image.bfloat16()
                elif args.precision == "fp16":
                    image = image.half()
                else:
                    image = image.float()

                image_list_sam.append(image)

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image_path, "max_pixels": args.max_pixels},
                    {"type": "text", "text": cur_prompt}
                ]}
            ]

        if args.prompt_for_seg == 1:
            messages += [{"role": "assistant", "content": [
                {"type": "text", "text": "Sure, [SEG]."}  # teacher forcing
                ]}
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        inputs = dict_to_cuda(inputs)
        input_ids = inputs['input_ids']

        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        pixel_values = inputs['pixel_values'].bfloat16() if 'pixel_values' in inputs else None
        pixel_values_videos = inputs['pixel_values_videos'].bfloat16() if 'pixel_values_videos' in inputs else None
        image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
        video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
        second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None

        if args.prompt_for_seg == 1:
            # It only allows for generating segmentation with forced prompting of Sure, SEG. Cant be used with other options
            image_sam = torch.stack(image_list_sam, dim=1)
            output_ids, pred_masks = model.evaluate(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                image_sam,
                resize_list,
                original_size_list,
            )
            response_text = 'Sure, [SEG].'

            if args.preds_dir != "":
                final_mask = pred_masks[0][0]
                final_mask = final_mask.detach().cpu().numpy()

                image = np.array(Image.open(image_path))
                image_vis = image.copy()
                image_vis[final_mask] = (255,0,0)

                cv2.imwrite(os.path.join(args.preds_dir, f"%05d.png"%photo_id), final_mask*255)
                cv2.imwrite(os.path.join(args.viz_dir, f"%05d.png"%photo_id), image_vis[:,:,::-1])
        else:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
        print(response_text)
        if args.prompt_for_seg == 3:
            # CVBench Format
            if args.postproc == "simple":
                response_text = response_text.strip().replace("addCriterion\n", "")
            ans_file.write(json.dumps({"question_id": photo_id,
                       "prompt": cur_prompt,
                       "answer": response_text.strip(),
                       "gt_answer": row["Correct Answer"],
                       "model_id": "RGA",
                       "text_options": line["text_options"]
                       }) + "\n")

        else:
            # GPT Grader Format
            if args.postproc == "simple":
                response_text = response_text.strip().replace("addCriterion\n", "")
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": photo_id,
                                       "prompt": cur_prompt,
                                       "answer": row["Correct Answer"],
                                       "response": response_text,
                                       "answer_id": ans_id,
                                       "model_id": "RGA",
                                       }) + "\n")
        ans_file.flush()
        torch.cuda.empty_cache()

    ans_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
