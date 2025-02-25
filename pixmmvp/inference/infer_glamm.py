import pandas as pd
import re
import cv2
import json
import bleach
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
import shortuuid
from PIL import Image
import torch.backends.cudnn as cudnn
import random

from eval.utils import *
from eval.ddp import *
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - GCG")

    parser.add_argument('--root', default=None, type=str)
    parser.add_argument('--root_images', default=None, type=str)
    parser.add_argument('--answers_file', default='answer.jsonl', type=str)

    parser.add_argument('--prompt_for_seg', default=0, type=int)
    parser.add_argument('--preds_dir', default="", type=str)
    parser.add_argument('--viz_dir', default="", type=str)
    parser.add_argument("--question_extension", type=str, default="Answer with the option's letter from the given choices directly.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--hf_model_path", required=True, help="The model path in huggingface format.")
    parser.add_argument("--img_dir", required=False, default="./data/GranDf/GranDf_HA_images/val_test",
                        help="The directory containing images to run inference.")
#    parser.add_argument("--output_dir", required=True, help="The directory to store the response in json format.")

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument(
        '--seed',
        type=int,
        default=2024,
        help='Random seed for reproducible text generation')

    return parser.parse_args()


def inference(instructions, image_path):
    # Filter out special chars
    instructions = bleach.clean(instructions)
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # Prepare prompt for model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(image_clip, image, input_ids, resize_list, original_size_list,
                                            max_tokens_new=512, bboxes=bboxes)#, temperature=args.temperature,
                                            #top_p=args.top_p)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    return cleaned_str, pred_masks, phrases


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]

    return image_id, image_path

def show_mask_pred(image, masks, save_dir='./output.png'):

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    final_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]
        final_mask[mask==1] = 1

    image = np.array(image)[:,:,::-1]
    overlay_img = image * 0.5 + _mask_image * 0.5
    overlay_img = overlay_img.astype(np.uint8)
    image[final_mask==1] = overlay_img[final_mask==1]
    image = Image.fromarray(image)
    image.save(save_dir)

    return

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


if __name__ == "__main__":
    args = parse_args()
#    init_distributed_mode(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {"torch_dtype": torch_dtype}
    model = GLaMMForCausalLM.from_pretrained(args.hf_model_path, low_cpu_mem_usage=True,
                                             seg_token_idx=seg_token_idx, **kwargs)
    # Update model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Initialize Global Image Encoder (CLIP)
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().cuda()  # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    answers_file = os.path.expanduser(args.answers_file)
    if os.path.dirname(answers_file):
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    benchmark_dir = os.path.join(args.root, 'Questions.csv')
    # Load and read the CSV
    df = pd.read_csv(benchmark_dir)  # Assuming the fields are separated by tabs
    df_objects = pd.read_csv(os.path.join(args.root, 'Objects.csv'))

    if args.preds_dir != "":
        if not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)
        if not os.path.exists(args.preds_dir):
            os.mkdir(args.preds_dir)

    # Iterate over all the images, run inference and save results
    for index, row in tqdm(df.iterrows()):

        # Construct the 'prompts' string
        photo_id = index+1
        image_path = os.path.join(args.root_images, f"{photo_id}.jpg")

        _, Object = df_objects.iloc[index]

        cur_prompt = row['Question'] + " " + row['Options']
        if args.prompt_for_seg == 1:
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

        output_text, pred_masks, _ = inference(cur_prompt, image_path)  # GLaMM Inference
        if args.prompt_for_seg == 3:
            # CVBench Format
            ans_file.write(json.dumps({"question_id": photo_id,
                       "prompt": cur_prompt,
                       "answer": output_text.strip(),
                       "gt_answer": row["Correct Answer"],
                       "model_id": "GLAMM",
                       "text_options": line["text_options"]
                       }) + "\n")

        else:
            # GPT Grader Format
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": photo_id,
                                       "prompt": cur_prompt,
                                       "answer": row["Correct Answer"],
                                       "response": output_text.strip(),
                                       "answer_id": ans_id,
                                       "model_id": "GLAMM",
                                       }) + "\n")
        ans_file.flush()

        # Convert the predicted masks into RLE format
        if args.preds_dir != "":
            pred_masks_tensor = pred_masks[0].cpu()
            binary_pred_masks = pred_masks_tensor > 0

            image_np = cv2.imread(image_path)
            if binary_pred_masks.shape[0] == 0:
                binary_pred_masks = torch.zeros((1, 1, *image_np.shape[:2])).float()
            else:
                binary_pred_masks = binary_pred_masks.unsqueeze(0).float()

            masks = F.interpolate(binary_pred_masks, size=image_np.shape[:2], mode='bilinear', align_corners=False)
            masks = masks.sigmoid() > 0.5
            masks = masks.to(torch.uint8).cpu().numpy()[:, 0]
            cv2.imwrite(os.path.join(args.preds_dir, "output%05d.png"%(index+1)), masks[0]*255)

            show_mask_pred(image_np, masks, save_dir='%s/output%05d.png'%(args.viz_dir, index+1))

    ans_file.close()
