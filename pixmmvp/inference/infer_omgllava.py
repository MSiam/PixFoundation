# Copyright (c) OpenMMLab. All rights reserved.
import json
import argparse
import copy
import os
import os.path as osp
import re
import sys
import random
import numpy as np
import torch.backends.cudnn as cudnn
import transformers
import cv2

import mmengine
import shortuuid
from tqdm import tqdm
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

import argparse
import os.path as osp
import torch.nn.functional as F
from PIL import Image

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

from xtuner.engine.hooks.evaluate_chat_hook import EvaluateChatHook

def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")

    parser.add_argument('--root', default=None, type=str)
    parser.add_argument('--root_images', default=None, type=str)
    parser.add_argument('--answers_file', default='answer.jsonl', type=str)

    parser.add_argument('--prompt_for_seg', default=0, type=int)
    parser.add_argument('--preds_dir', default="", type=str)
    parser.add_argument('--viz_dir', default="", type=str)

    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default="internlm2_chat",
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,#0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=2024,
        help='Random seed for reproducible text generation')
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--model_path", type=str, default="./pretrained/")
    args = parser.parse_args()
    return args


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
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

def give_options(input_string):
    parts = input_string.split("(")
    result = [part.split(")")[1].strip() for part in parts[1:]]
    return result

def modify_paths(cfg, path):
    if type(cfg) != mmengine.config.config.ConfigDict:
        return cfg

    for key in cfg.keys():
        if type(cfg[key]) == str and 'pretrained' in cfg[key]:
            cfg[key] = cfg[key].replace('pretrained', path)
        if type(cfg[key]) == mmengine.config.config.ConfigDict:
            cfg[key] = modify_paths(cfg[key], path)
    return cfg

def main():
    args = parse_args()
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
    if 'LLaVAModel' or 'OMG' in model_name:
        cfg.model.pretrained_pth = None

    model_path_keys = ['llm_name_or_path', 'pretrained_pth', 'omg_ov_class_embed_path', 'omg_head_pretrain_pth_path']
    for key in model_path_keys:
        cfg[key] = os.path.join(args.model_path, cfg[key])
    cfg.model = modify_paths(cfg.model, args.model_path)
    model = BUILDER.build(cfg.model)
    print(model.state_dict().keys())

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    image_processor = cfg.image_processor
    image_processor_type = image_processor['type']
    del image_processor['type']
    image_processor = image_processor_type(**image_processor)

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }
    if False:
        pass
    else:
        if args.with_plugins is None:
            inner_thoughts_open = False
            calculate_open = False
            solve_open = False
            search_open = False
        else:
            assert args.prompt_template == args.system_template == 'moss_sft'
            from plugins import plugins_api
            inner_thoughts_open = True
            calculate_open = 'calculate' in args.with_plugins
            solve_open = 'solve' in args.with_plugins
            search_open = 'search' in args.with_plugins
            # pre-import for api and model preparation
            if calculate_open:
                from plugins import calculate  # noqa: F401
            if solve_open:
                from plugins import solve  # noqa: F401
            if search_open:
                from plugins import search  # noqa: F401
        # build llm
        llm = model.llm
        tokenizer = model.tokenizer

        model.cuda()
        model.eval()
        llm.eval()
        visual_encoder = model.visual_encoder
        projector = model.projector
        projector_text2vision = model.projector_text2vision

        stop_words = args.stop_words
        sep = ''
        if args.prompt_template:
            template = PROMPT_TEMPLATE[args.prompt_template]
            stop_words += template.get('STOP_WORDS', [])
            sep = template.get('SEP', '')
        stop_criteria = get_stop_criteria(
            tokenizer=tokenizer, stop_words=stop_words)

        if args.no_streamer:
            streamer = None
        else:
            streamer = TextStreamer(tokenizer, skip_prompt=True)

        gen_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            #top_k=args.top_k,
            #repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

        answers_file = os.path.expanduser(args.answers_file)
        # Check if the directory is specified in the path
        if os.path.dirname(answers_file):
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")

        benchmark_dir = os.path.join(args.root, 'Questions.csv')
        # Load and read the CSV
        df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
        df_objects = pd.read_csv(os.path.join(args.root, 'Objects.csv'))

        if args.viz_dir != "" and not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)
        if args.preds_dir != "" and not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)

        for index, row in tqdm(df.iterrows()):

            # Construct the 'prompts' string
            photo_id = index+1
            image_path = os.path.join(args.root_images, f"{photo_id}.jpg")

            image = load_image(image_path)


            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean))
            image_for_show = image
            image = image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)

            visual_outputs = visual_encoder(image, output_hidden_states=True)
            print([item.shape for item in visual_outputs])
            pixel_values = projector(visual_outputs)

            _, Object = df_objects.iloc[index]
            Object.strip()

            if args.prompt_for_seg == 0:
                cur_prompt = row['Question'] + " " + row['Options']
            elif args.prompt_for_seg == 1:
                cur_prompt = f'Can you please segment {Object} in the given image'
            elif args.prompt_for_seg == 2:
                cur_prompt += f' Can you also please segment {Object} in the given image'
            elif args.prompt_for_seg == 3:
                line = {
                        "question": str(row[1]),
                        "imageId": int(row[0])-1,
                        "options": str(row[2]),
                        "text_options": give_options(str(row[2])),
                        "answer": str(row[3])
                       }

                cur_prompt = process(line, args)

            text = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
            print(text)

            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=text, round=1, bot_name=args.bot_name)
            inputs = prompt_text

            chunk_encode = []
            for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0:
                    cur_encode = tokenizer.encode(chunk)
                else:
                    cur_encode = tokenizer.encode(
                        chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    ids.append(IMAGE_TOKEN_INDEX)
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=llm, input_ids=ids, pixel_values=pixel_values)
            print(mm_inputs['inputs_embeds'].shape)
            # mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(torch.float16)

            generate_output = llm.generate(
                **mm_inputs,
                generation_config=gen_config,
                streamer=streamer,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            if args.preds_dir != "":
                # Parse segmentation output and save
                hidden_states = generate_output.hidden_states
                last_hidden_states = [item[-1][0] for item in hidden_states]
                last_hidden_states = torch.cat(last_hidden_states, dim=0)
                seg_hidden_states = get_seg_hidden_states(
                    last_hidden_states, generate_output.sequences[0][:-1],
                    # last_hidden_states, generate_output.sequences[0],
                    seg_id=model.seg_token_idx
                )

                if len(seg_hidden_states) != 0:
                    seg_hidden_states = projector_text2vision(seg_hidden_states)
                    batch_idxs = torch.zeros((seg_hidden_states.shape[0], ),
                                              dtype=torch.int64).to(seg_hidden_states.device)
                    pred_masks_list = model.visual_encoder.forward_llm_seg(seg_hidden_states, batch_idxs)
                    print((pred_masks_list[-1].flatten(2) > 0).sum(-1))
                    print(pred_masks_list[-1].shape)

                    show_mask_pred(image_for_show, pred_masks_list[-1], save_dir='%s/output%05d.png'%(args.viz_dir, index+1))

                    masks = pred_masks_list[-1]
                    masks = F.interpolate(masks, size=image_for_show.size, mode='bilinear', align_corners=False)
                    masks = masks.sigmoid() > 0.5
                    masks = masks.to(torch.uint8).cpu().numpy()[0, 0]
                    cv2.imwrite(os.path.join(args.preds_dir, "output%05d.png"%(index+1)), masks*255)
                else:
                    masks = np.zeros((image_for_show.size), dtype=np.uint8)
                    cv2.imwrite(os.path.join(args.preds_dir, "output%05d.png"%(index+1)), masks*255)

            output_text = tokenizer.decode(generate_output.sequences[0])
            output_text = output_text.replace("\n", "").replace("  ", " ").replace("<s>", "").replace("<|im_end|>", '')
            output_text = output_text.replace('[SEG]', '').replace('<p>', '').replace('</p>', '')

            ans_id = shortuuid.uuid()

            if args.prompt_for_seg == 3:
                # CVBench Format
                ans_file.write(json.dumps({"question_id": photo_id,
                           "prompt": cur_prompt,
                           "answer": output_text.strip(),
                           "gt_answer": row["Correct Answer"],
                           "model_id": model_name,
                           "text_options": line["text_options"]
                           }) + "\n")

            else:
                # GPT Grader Format
                ans_file.write(json.dumps({"question_id": photo_id,
                                           "prompt": cur_prompt,
                                           "answer": row["Correct Answer"],
                                           "response": output_text.strip(),
                                           "answer_id": ans_id,
                                           "model_id": model_name,
                                           }) + "\n")
            ans_file.flush()

        ans_file.close()

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    print(output_ids)
    return hidden_states[-n_out:][seg_mask]

def show_mask_pred(image, masks, save_dir='./output.png'):

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    masks = F.interpolate(masks, size=image.size, mode='bilinear', align_corners=False)
    masks = masks.sigmoid() > 0.5
    masks = masks.to(torch.uint8).cpu().numpy()[:, 0]

    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    final_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]
        final_mask[mask==1] = 1

    image = np.array(image)
    overlay_img = image * 0.5 + _mask_image * 0.5
    overlay_img = overlay_img.astype(np.uint8)
    image[final_mask==1] = overlay_img[final_mask==1]
    image = Image.fromarray(image)
    image.save(save_dir)

    return

if __name__ == '__main__':
    main()
