import argparse
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import random
import string
from pixf_utils.prompt_sens_utils  import guided_crop, guided_maskout
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import base64
import copy

from pixmmvp.dataset.register_mmvp import register_new_dataset
from pixmmvp.dataset.custom_coco_dataset import CustomCOCODataset

import torch.utils.data as torchdata
from tqdm import tqdm as tqdm
from detectron2.data.build import trivial_batch_collator
import cv2
import torch
from PIL import Image

def denormalize(img, mean, scale):
    img = torch.tensor(img)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

def overlay_mask(img, mask):
    def PIL2array(img):
        return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    mask_color[mask==1, 1]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==1,:] = blended_arr[mask==1,:]
    return img2

def visualize_masks(img, masks):
    for mask in masks:
        img = overlay_mask(img, mask)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--n_variations_per_type", type=int, default=10)
    parser.add_argument("--stop_at", type=int, default=29)
    parser.add_argument("--start_at", type=int, default=0)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    variation_type = 0
    batch_size = 8
    num_workers = 0
    register_new_dataset(args.root)

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    dataset = CustomCOCODataset(mean, std)

    # Dataloading
    dataloader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                      drop_last=False, num_workers=num_workers,
                                      collate_fn=trivial_batch_collator)

    with open(os.path.join(args.root, 'Segmentations.json')) as f:
        json_info = json.load(f)

    curr_id = 0
    segmentations_variations = {}
    for var_i in range(args.n_variations_per_type*2):
        if var_i % args.n_variations_per_type == 0:
            variation_type += 1
        print("####################### Variation Type ", variation_type, ', var #', var_i)
        if var_i > args.stop_at:
            break
        if var_i < args.start_at:
            continue

        if var_i not in segmentations_variations:
            segmentations_variations[var_i] = []

        np.random.seed(2024+var_i)
        random.seed(2024+var_i)

        for minibatch in tqdm(dataloader):
            curr_id += 1

            out_img_dir = os.path.join(args.out_dir, 'var_%02d'%var_i, 'MMVP Images/')
            if not os.path.exists(out_img_dir):
                os.makedirs(out_img_dir)

            for i in range(len(minibatch)):
                filename = minibatch[i]['file_name']
                img = copy.deepcopy(minibatch[i]['image'])
                anno_masks = [copy.deepcopy(m) for m in minibatch[i]['masks']]
                curr_img_id = minibatch[i]['image_id']

                img = np.array(img.permute(1,2,0))
                img = denormalize(img, mean, std)

                viz_img = visualize_masks(img, anno_masks)

                if variation_type == 1:
                    img, mask = guided_crop(img, anno_masks[0])
                else:
                    img, mask = guided_maskout(img, anno_masks[0])

                img = img[:,:,::-1]

                new_segmentation = mask_util.encode(np.asfortranarray(np.asarray(mask, np.uint8)))
                new_segmentation['counts'] = base64.b64encode(new_segmentation['counts']).decode('utf-8')
                if mask.sum() == 0:
                    new_box = []
                else:
                    new_box = list(cv2.boundingRect(mask))

                new_segmentation = {'segmentation': new_segmentation, 'box': new_box, 'area': int(mask.sum()),
                                    'iscrowd': 0, 'id': curr_id, 'image_id': curr_img_id, 'category_id': None}
                segmentations_variations[var_i].append(new_segmentation)

                cv2.imwrite(os.path.join(out_img_dir, filename.split('/')[-1]), img)

    # Save segmentations variations as json
    for var_i, segmentation_info in segmentations_variations.items():
        json_info_copy = json_info.copy()
        assert len(segmentation_info) >= len(json_info_copy['images']), "Wrong segmentation annotations"
        json_info_copy['annotations'] = segmentation_info

        with open(os.path.join(args.out_dir, 'Segmentations_%02d.json'%var_i), 'w') as f:
            json.dump(json_info_copy, f)
