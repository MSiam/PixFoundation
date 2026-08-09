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
from datasets import load_dataset
sys.path.append(".")

import difflib
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
    parser.add_argument("--cvbench_section", type=str, default="ADE20K")
    parser.add_argument("--data_file", type=str, default="test.parquet")
    parser.add_argument("--grounding_prompts_file", default="", type=str)
    parser.add_argument("--variation_idx", default=-1, type=int)
    return parser.parse_args(args)

def get_overlap(s1, s2):
    seq = difflib.SequenceMatcher(a=s1.lower(),b=s2.lower())
    return (seq.ratio() != 0)


def retrieve_higlighted_object(prompt, current_objects):
    # Get object annotated by red box
    token = prompt.split("(annotated by the red box)")[0]
    for obj in current_objects:
        if get_overlap(token, obj):
            return obj
    return None

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
    cvbench_dataset = load_dataset("parquet", data_files={"validation": os.path.join(args.dataset_root, "CV-Bench", args.data_file)})
    prompts_to_objects = pd.read_csv(os.path.join(args.dataset_root, "CV-Bench", 'Objects.csv'))
    prompts_to_objects = prompts_to_objects.set_index('Id').T.to_dict('list')
    for k, v in prompts_to_objects.items():
        if '(' in v[-1]:
            prompts_to_objects[k] = tuple([x.replace("'", "") for x in v[-1].replace('(','').replace(')','').split(',')])
        else:
            prompts_to_objects[k] = v[-1]

    answers_file = os.path.expanduser(args.answers_file)
    # Check if the directory is specified in the path
    if os.path.dirname(answers_file):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)


    ans_file = open(answers_file, "w")
    if args.preds_dir != "":
        if not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)
        if not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)

    if args.grounding_prompts_file != '':
        grounding_prompts = pd.read_csv(os.path.join(args.dataset_root, args.grounding_prompts_file),
                                        index_col=0)

    for idx, entry in tqdm(enumerate(cvbench_dataset['validation'])):
        if args.cvbench_section != "ALL" and args.cvbench_section != entry['source']:
            continue

        # Construct the 'prompts' string
        if entry['source'] == 'ADE20K':
            prefix = os.path.join(entry['source'], 'validation/images/')
        elif entry['source'] == 'COCO':
            prefix = os.path.join(entry['source'], 'coco/images/')
        else:
            continue

        source_filename = entry['source_filename']

        if entry['source'] == 'COCO':
            source_filename = entry['source_filename'].replace('/coco/val2017/', '')

        image_path = os.path.join(args.dataset_root, prefix, source_filename)
        Object = prompts_to_objects[entry['idx']]

        cur_prompt = entry['prompt']
        if args.prompt_for_seg == 1:
            if 'red box' in cur_prompt:
                highlighted_obj = retrieve_higlighted_object(cur_prompt, Object)
                for obj in Object:
                    if obj != highlighted_obj:
                        other_obj = obj
                Object = highlighted_obj + " (annotated by the red box) and " + other_obj
                print(cur_prompt)
                print('Filtered: ', Object)
            elif type(Object) == tuple:
                Object = Object[0] + ' and ' + Object[1]

            if args.variation_idx != -1 and grounding_prompts is not None:
                cur_prompt = grounding_prompts['Template'].iloc[args.variation_idx].format(Object)
            else:
                cur_prompt = f'Can you please segment {Object} in the given image'
        elif args.prompt_for_seg == 2:
            cur_prompt += f' Can you also please segment {Object} in the given image'
        elif args.prompt_for_seg == 3:
            cur_prompt += f"\n{args.question_extension}"

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

                cv2.imwrite(os.path.join(args.preds_dir, "%s.png"%source_filename.split('.')[0]), final_mask*255)
                cv2.imwrite(os.path.join(args.viz_dir, "%s/%s"%(args.viz_dir, source_filename)), image_vis[:,:,::-1])
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
            if args.postproc == "simple":
                response_text = response_text.strip().replace("addCriterion\n", "")
            ans_file.write(json.dumps({
                            "questionId": idx,
                            "image": entry["source_filename"],
                            "prompt": cur_prompt,
                            "answer": response_text.strip(),
                            "gt_answer": entry["answer"],
                            "category": entry["task"],
                            "options": entry["choices"],
                            "model_id": "RGA"
                            }) + "\n")
        else:
            if args.postproc == "simple":
                response_text = response_text.strip().replace("addCriterion\n", "")
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": entry['source_filename'],
                                       "prompt": cur_prompt,
                                       "answer": entry["answer"],
                                       "response": response_text.strip(),
                                       "answer_id": ans_id,
                                        "model_id": "RGA"
                                       }) + "\n")
        ans_file.flush()
        torch.cuda.empty_cache()

    ans_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
