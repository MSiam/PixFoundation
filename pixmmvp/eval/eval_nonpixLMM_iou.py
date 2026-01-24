import glob
import argparse
import numpy as np
from pixmmvp.dataset.register_mmvp import register_new_dataset
from pixmmvp.dataset.custom_coco_dataset import CustomCOCODataset
import torch.utils.data as torchdata
from tqdm import tqdm as tqdm
from detectron2.data.build import trivial_batch_collator
import cv2
import os
import torch
from PIL import Image
import pandas as pd
import json

def compute_iou(segmentation, annotation):
    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="PixMMVP")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--preds_dir", type=str)
    parser.add_argument("--type", type=str, default="original")
    parser.add_argument("--meta_file", type=str, default="")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--viz_dir", default="", type=str)
    parser.add_argument("--remove_none_flag", action="store_true")
    parser.add_argument("--varidx", type=int, default=-1)
    return parser.parse_args()

def retrieve_highest_iou(filename, preds_files, anno_masks):
    orig_filename = filename
    filename = filename.split('/')[-1].split('.')[0]
    matched_preds_files = [pf for pf in preds_files if filename == pf.split('/')[-1].split('_')[0]]

    best_iou = 0
    mask_area = 1e20
    best_matched_mask = None
    best_matched_file = None

    for predf in matched_preds_files:
        pred_mask = cv2.imread(predf, 0)
        pred_mask[pred_mask==255] = 1

#        if anno_masks.sum() == 0:
#            # retrieve the smallest mask since all will have iou = 0
#            if mask_area > pred_mask.sum():
#                mask_area = pred_mask.sum()
#                best_matched_mask = pred_mask
#                best_matched_file= predf
#            img = Image.open(orig_filename)
#            best_matched_mask = np.zeros((img.size[1], img.size[0]), np.uint8)
#        else:
        iou = compute_iou(pred_mask, anno_masks)
        if iou >= best_iou:
            best_iou = iou
            best_matched_mask = pred_mask
            best_matched_file = predf
    return best_matched_mask

def denormalize(img, mean, scale):
    img = torch.tensor(img)
    img = img * torch.tensor(scale) + torch.tensor(mean)
    img = img.cpu().numpy()
    img = np.asarray(img[:,:,::-1]*255, np.uint8)
    return img

def retrieve_max_spacy_score(filename, preds_files, meta_info):
    filename = filename.split('/')[-1].split('.')[0]
    matched_preds_files = [pf for pf in preds_files if filename == pf.split('/')[-1].split('_')[0]]
    matched_preds_files= sorted(matched_preds_files)

    best_spacy_score = 0
    best_matched_mask = None
    best_matched_file = None

    for predf in matched_preds_files:
        pred_mask = cv2.imread(predf, 0)
        pred_mask[pred_mask==255] = 1

        tokens = predf.split('/')[-1].split('_')
        phrase_idx = int(tokens[1])

        _, _, spacy_score = meta_info[filename+'.jpg'][phrase_idx]
        if spacy_score >= best_spacy_score :
            best_spacy_score = spacy_score
            best_matched_mask = pred_mask
            best_matched_file = predf

    spacy_threshold = 0.7
    if best_spacy_score < spacy_threshold and best_matched_mask is not None:
        best_matched_mask = np.zeros_like(best_matched_mask)
    return best_matched_mask

def retrieve_auto_select(filename, preds_dir, preds_files, meta_info):

    for row in meta_info:
        fname = str(int(row["question_id"])+1)+'.jpg'
        if fname == filename.split('/')[-1]:
            selectedf = os.path.join(preds_dir, row["selected file"])
            if selectedf.split('/')[-1] == 'NONE':
                # Object doesnt exist in image
                img = Image.open(filename)
                return np.zeros((img.size[1], img.size[0]), np.uint8)

            pred_mask = cv2.imread(selectedf, 0)
            pred_mask[pred_mask==255] = 1
            return pred_mask

    return None

def load_meta_info(meta_file):
    meta_info = {}
    with open(meta_file, "r") as f:
        for line in f:
            tokens = line.split('- ')
            fname = tokens[0].strip()
            if fname not in meta_info:
                meta_info[fname] = []

            phrase, core_word, spacy_score = tokens[2].strip().split('| ')
            meta_info[fname].append((phrase.strip(), core_word.strip(), float(spacy_score.strip()) ))
    return meta_info

def load_meta_json_info(answer_file):
    datas = []
    with open(answer_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            datas.append(data)
    return datas

def show_mask_pred(image, mask):
    color = (0, 0, 255)
    _mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
    _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
    _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

    image = np.array(image)
    overlay_img = image * 0.5 + _mask_image * 0.5
    overlay_img = overlay_img.astype(np.uint8)
    image[mask==1] = overlay_img[mask==1]
    return image

if __name__ == "__main__":
    args = parse_args()

    register_new_dataset(args.dataset_root, args.varidx)

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    if args.varidx == -1:
        prefix = 'mmvp_val'
    else:
        prefix = 'mmvp_sensitivity_val'

    dataset = CustomCOCODataset(mean, std, prefix, args.varidx)

    # Dataloading
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batchsize,
                                      drop_last=False, num_workers=args.workers,
                                      collate_fn=trivial_batch_collator)

    preds_files = sorted(glob.glob(os.path.join(args.preds_dir, '*.png')))
    if len(preds_files) == 0:
        raise Exception("No segmentation output")

    miou = 0
    nimages = 0
    mean = [123.675, 116.28, 103.53]
    scale = [58.395, 57.12, 57.375]

    if args.type == "spacy_score":
        assert os.path.exists(args.meta_file), "Meta information with Spacy scores file is missing"
        meta_info = load_meta_info(args.meta_file)
    elif args.type == "auto":
        meta_info = load_meta_json_info(args.meta_file)

    if args.type in ["spacy_score", "oracle", "auto"]:
        if args.viz_dir != "" and not os.path.exists(args.viz_dir):
            os.mkdir(args.viz_dir)

    if args.remove_none_flag:
        none_flag_file = open(os.path.join(args.dataset_root, 'meta_none.txt'), 'r')
        none_flag_images = none_flag_file.read().strip().split('\n')

    for idx, minibatch in tqdm(enumerate(dataloader)):
        for i in range(len(minibatch)):

            filename = minibatch[i]['file_name']
            if args.remove_none_flag:
                if filename.split('/')[-1] in none_flag_images:
                    continue

            anno_masks = minibatch[i]['masks']
            if type(anno_masks) == list:
                assert len(anno_masks) == 1, "More than one mask Issue"
                anno_masks = np.array(anno_masks[0], dtype=np.uint8)

            if args.type == "original":
                # Evaluate on the original predictions per frame
                pred_mask = cv2.imread(preds_files[idx * args.batchsize + i], 0)
                pred_mask[pred_mask==255] = 1
            elif args.type == "oracle":
                if anno_masks.sum() == 0:
                    img = Image.open(filename)
                    pred_mask = np.zeros((img.size[1], img.size[0]), np.uint8)
                else:
                    pred_mask = retrieve_highest_iou(filename, preds_files, anno_masks)
            elif args.type == "spacy_score":
               pred_mask = retrieve_max_spacy_score(filename, preds_files, meta_info)
            elif args.type == "auto":
                pred_mask = retrieve_auto_select(filename, args.preds_dir, preds_files, meta_info)

            if pred_mask is not None:
                iou = compute_iou(pred_mask, anno_masks)

                if args.type in ["spacy_score", "oracle", "auto"] and args.viz_dir != "":
                    img = minibatch[i]['image']
                    img = np.array(img.permute(1,2,0))
                    img = denormalize(img, mean, scale)
                    vis_image = show_mask_pred(img, pred_mask)
                    cv2.imwrite(os.path.join(args.viz_dir, filename.split('/')[-1]), vis_image)
            else:
                iou = 0
                if args.type in ["spacy_score", "oracle", "auto"] and args.viz_dir != "":
                    img = minibatch[i]['image']
                    img = np.array(img.permute(1,2,0))
                    img = denormalize(img, mean, scale)
                    cv2.imwrite(os.path.join(args.viz_dir, filename.split('/')[-1]), img)

            miou += iou
            nimages += 1
    print("Total mIoU = ", float(miou / nimages))
    print("# Images = ", nimages)
