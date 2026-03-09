#####################################################################
#                           PixFoundation
#       Interpretability of MLLMs on When Grounding Emerges
#       => code can run standalone w/o this repo
#####################################################################
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import random
from PIL import Image
import os
import sys
import cv2
import matplotlib.pyplot as plt
import argparse
sys.path.append("../LLaVA/")

from transformers import AutoTokenizer
from segment_anything import sam_model_registry, SamPredictor
import spacy
import openai
import base64

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

def get_spacy_embedding(phrase, spacy_model):
    phrase = phrase.lower()
    if phrase.startswith('the '):
        phrase = phrase[4:]
    elif phrase.startswith('an '):
        phrase = phrase[3:]
    elif phrase.startswith('a '):
        phrase = phrase[2:]
    doc = spacy_model(phrase)
    embedding = torch.tensor(doc.vector)
    return embedding

def group_tokens(tokens, tokenizer, spacy_model):
    # find correspondence between tokens and chars
    token_start = []
    token_end = []
    cur_length = 0
    for i in range(len(tokens)):
        token_start.append(cur_length)
        sequence = tokenizer.decode(tokens[:i+1], skip_special_tokens=True)
        cur_length = len(sequence)
        token_end.append(cur_length)

    # find noun phrases
    phrases = []
    coreword_start = []
    coreword_end = []
    phrase_start= []
    phrase_end = []
    core_words = []
    doc = spacy_model(sequence)
    for np in doc.noun_chunks:
        phrases.append(np.text)
        core_words.append(np.root.text)
        coreword_start.append(np.root.idx)
        coreword_end.append(np.root.idx + len(np.root.text))
        phrase_start.append(np.start_char)
        phrase_end.append(np.start_char + len(np.text))

    # group tokens
    groups = []
    for i in range(len(phrases)):
        group_tokens = []
        for j in range(len(tokens)):
            # check if token has overlap with phrase
            if token_start[j] < phrase_end[i] and token_end[j] > phrase_start[i]:
                group_tokens.append(j)
        group = {
            'phrase': phrases[i],
            'core_word': core_words[i],
            'tokens': group_tokens,
            'start_char': coreword_start[i],
            'end_char': coreword_end[i],
            'phr_start_char': phrase_start[i],
            'phr_end_char': phrase_end[i]
        }
        groups.append(group)

    return groups

def encode_image_to_base64(image_path):
    if type(image_path) != str:
        image_path.save('temp.jpg')
        image_path = 'temp.jpg'

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

NUM_SECONDS_TO_SLEEP = 10
def mask_selection(prompt, urls, ref_expr):
    prompt = prompt.format(ref_expr, len(urls))

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
                model='gpt-5.2',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the segmentation.'
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
    best_idx = int(answer)-1
    return best_idx, answer


def inference(prompt, image_path, model, tokenizer):
    conv_mode = 'llava_v1'

    qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            #image_sizes=[image.size],
            do_sample=True,
            temperature=0.2,
            top_p=None,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            output_attentions=True,
            return_dict_in_generate=True)

    input_token_len = input_ids.shape[1]
    answer = tokenizer.batch_decode(output_ids['sequences'][:, input_token_len:], skip_special_tokens=True)[0]
    print('Output: ', answer)
    return answer, input_ids, output_ids

def interpretability_mllm_grounding(image_path, input_ids, output_ids, sam_predictor, spacy_model):
    reg_tokens = 0
    feature_height = feature_width = 24

    input_token_len = input_ids.shape[1]
    image_token_start_index = -1
    for i in range(input_ids.shape[1]):
        if input_ids[0, i] == IMAGE_TOKEN_INDEX:
            image_token_start_index = i
            break
    assert image_token_start_index >= 0

    image = Image.open(image_path).convert('RGB')
    image_width = image.width
    image_height = image.height

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
            if reg_tokens > 0:
                image_token_length -= reg_tokens
                image_token_start_index += reg_tokens
            image_token_end_index = image_token_start_index + image_token_length
            assert image_token_length == feature_height * feature_width, \
                f'Image token length mismatch: Expected {feature_height * feature_width}, got {image_token_length}'
        save_attn_i = save_attn_i[:, :, -1, image_token_start_index:image_token_end_index]  # n_layers x n_heads x n_image_tokens
        save_attn_i = save_attn_i.mean(dim=(0, 1))                                          # n_image_tokens
        save_attn_i = save_attn_i.reshape(feature_height, feature_width)          # feature_height x feature_width
        save_attn.append(save_attn_i.detach().cpu())

    save_attn = torch.stack(save_attn)

    attentions = save_attn.float()
    attn_mean = attentions.mean(dim=0)
    attentions = attentions - attn_mean

    # group tokens
    groups = group_tokens(save_sequences[0], tokenizer, spacy_model)
    if len(groups) == 0:
        print('Ended with zero groups!')

    if len(groups) != 0:
        for group in groups:
            attns = attentions[group['tokens']]
            attn = attns.mean(dim=0)
            group['attention'] = attn

        # create segmentation masks
        group_scores = [group['attention'] for group in groups]
        group_scores = torch.stack(group_scores)

        upsample_size = max(image_height, image_width)
        crop_h_start = (upsample_size - image_height) // 2
        crop_h_end = crop_h_start + image_height
        crop_w_start = (upsample_size - image_width) // 2
        crop_w_end = crop_w_start + image_width
        upsample_scores = torch.nn.functional.interpolate(group_scores.unsqueeze(0),
                                                        size=(upsample_size, upsample_size),
                                                        mode='bicubic', align_corners=False).squeeze(0)
        upsample_scores = upsample_scores[:, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

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
            mask = pred_masks[group_index]
            groups[group_index]['mask'] = mask
            groups[group_index]['point'] = point_coords_np[group_index]
            keep_groups.append(group_index)
            color = np.random.randint(0, 256, 3)
            groups[group_index]['color'] = color
            area = mask.sum()
            groups[group_index]['area'] = area

        groups = [groups[i] for i in keep_groups]
        pred_masks = pred_masks[keep_groups]

        images_vis= []
        strings = []
        for gridx, group in enumerate(groups):
            masks = group['mask']
            strings.append("%d - %s| %s"%(gridx, group['phrase'], group['core_word']))
            print(strings[-1])

            for midx, mask in enumerate(masks):
                image_vis = np.array(image).copy()
                image_vis[mask] = (255,0,0)
                image_vis = cv2.circle(image_vis, point_coords_np[gridx], 5, (0,0,0), -1)
                image_vis = Image.fromarray(image_vis)
                images_vis.append(image_vis)

    return images_vis, strings

def iterative_mask_selection(images_vis, ref_expr):
    prompt = "Select the image that has {} best highlighted in red color than the others? Answer with a number from 1 to {}. Mention the number only."

    MAX_IMGS = 12
    if len(images_vis) < MAX_IMGS:
        final_imgs = images_vis
        final_indices = range(len(images_vis))
    else:
        parts = range(len(images_vis) // MAX_IMGS + 1)
        final_imgs = []
        final_indices = []
        for part in parts:
            if part == parts[-1]:
                curr_imgs = images_vis[part*MAX_IMGS:]
            else:
                curr_imgs = images_vis[part*MAX_IMGS:(part+1)*MAX_IMGS]

            try:
                print('Prompt: ', prompt)
                selected_index, answer = mask_selection(prompt, curr_imgs, ref_expr)
                final_indices.append(part*MAX_IMGS + selected_index)
                final_imgs.append(curr_imgs[selected_index])
                print('Answer: ', answer)
            except:
                selected_index = -1
                print("Error parsing the answer ", answer)

    try:
        print('Prompt: ', prompt)
        selected_index, answer = mask_selection(prompt, final_imgs, ref_expr)
        print('Answer: ', answer)
    except:
        selected_index = -1
        print("Error parsing the answer ", answer)
    return final_indices[selected_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='image1.jpeg')
    parser.add_argument('--ref_expr', type=str, default="kitten's closed eyes")
    parser.add_argument('--openai_api_key', type=str)
    args = parser.parse_args()

    openai.api_key = args.openai_api_key
    cudnn.benchmark = False
    cudnn.deterministic = True
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load models
    model_path = 'liuhaotian/llava-v1.5-7b'
    sam_model = 'vit_h'
    sam_ckpt = 'sam_vit_h_4b8939.pth'

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model.get_vision_tower().to(dtype=torch.float16)

    sam_model = sam_model_registry[sam_model](checkpoint=sam_ckpt).cuda()
    sam_predictor = SamPredictor(sam_model)
    spacy_model = spacy.load('en_core_web_lg')

    # Set Image and referring expression
    image_path = args.image_path
    ref_expr = args.ref_expr
    prompt = f"Identify {ref_expr} in the scene."
    print('Input Prompt: ', prompt)

    answer, input_ids, output_ids = inference(prompt, image_path, model, tokenizer)
    images_vis, strings = interpretability_mllm_grounding(image_path, input_ids, output_ids, sam_predictor, spacy_model)

    selected_index = iterative_mask_selection(images_vis, ref_expr)

    ncols = 3
    nrows = len(images_vis) //ncols
    if nrows > 10:
        nrows = 10
        group_images_vis = group_images_vis[:nrows*ncols]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))

    for i in range(nrows):
        for j in range(ncols):
            if i*ncols+j == selected_index:
                for spine in axes[i, j].spines.values():
                    spine.set_edgecolor('blue')
                    spine.set_linewidth(3)

            axes[i, j].imshow(images_vis[i*ncols+j])

    plt.show()
