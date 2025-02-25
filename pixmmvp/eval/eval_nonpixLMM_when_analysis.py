import glob
import argparse
import numpy as np
from dataset_helpers.register_mmvp import register_new_dataset
from dataset_helpers.custom_coco_dataset import CustomCOCODataset
import torch.utils.data as torchdata
from tqdm import tqdm as tqdm
from detectron2.data.build import trivial_batch_collator
import cv2
import os
import torch
from PIL import Image
import pandas as pd

def compute_iou(segmentation, annotation):
    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        if annotation.sum() == 0:
            import pdb; pdb.set_trace()
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(description="PixMMVP")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--when_out_dir", type=str)
    parser.add_argument("--preds_dir", type=str)
    parser.add_argument("--meta_file", type=str, default="")
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    return parser.parse_args()

def retrieve_highest_iou(filename, preds_files, anno_masks):
    filename = filename.split('/')[-1].split('.')[0]
    matched_preds_files = [pf for pf in preds_files if filename == pf.split('/')[-1].split('_')[0]]

    best_iou = 0
    best_matched_mask = None
    best_matched_file = None
    best_meta_info = None

    for predf in matched_preds_files:
        pred_mask = cv2.imread(predf, 0)
        pred_mask[pred_mask==255] = 1

        tokens = predf.split('/')[-1].split('_')
        phrase_idx = int(tokens[1])

        iou = compute_iou(pred_mask, anno_masks)
        if iou >= best_iou:
            best_iou = iou
            best_matched_mask = pred_mask
            best_matched_file = predf
            try:
                best_meta_info = meta_info[filename+'.jpg'][phrase_idx]
            except:
                break

    return best_matched_mask, best_meta_info

def load_meta_info(meta_file):
    meta_info = {}
    line_idx = 0
    with open(meta_file, "r") as f:
        for line in f:
            tokens = line.split('- ')
            fname = tokens[0].strip()
            if fname not in meta_info:
                meta_info[fname] = []
            phrase, core_word, spacy_score, phrase_st, phrase_end, text = tokens[2].strip().split('| ')
            meta_info[fname].append((phrase.strip(), core_word.strip(), float(spacy_score.strip()),
                                    int(phrase_st), int(phrase_end), text))

            line_idx += 1

    return meta_info

if __name__ == "__main__":
    args = parse_args()

    assert args.batchsize == 1, "Need to operate on image by image to retrieve the correct object"
    register_new_dataset(args.dataset_root)

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    dataset = CustomCOCODataset(mean, std)

    # Dataloading
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batchsize,
                                      drop_last=False, num_workers=args.workers,
                                      collate_fn=trivial_batch_collator)

    preds_files = sorted(glob.glob(os.path.join(args.preds_dir, '*.png')))
    if len(preds_files) == 0:
        raise Exception("No segmentation output")

    miou = 0
    nimages = 0

    assert os.path.exists(args.meta_file), "Meta information with Spacy scores file is missing"
    meta_info = load_meta_info(args.meta_file)

    hist_token_loc = []
    token_loc = []
    token_concept = []

    if not os.path.exists(args.when_out_dir):
        os.mkdir(args.when_out_dir)

    df_objects = pd.read_csv(os.path.join(args.dataset_root, 'Objects.csv'))

    for idx, minibatch in tqdm(enumerate(dataloader)):
        _, Object = df_objects.iloc[idx]

        if Object.strip() == "None":
            continue

        filename = minibatch[0]['file_name']
        #if filename.split('/')[-1] != '292.jpg':
        #    continue
        anno_masks = minibatch[0]['masks']
        if type(anno_masks) == list:
            assert len(anno_masks) == 1, "More than one mask Issue"
            anno_masks = np.array(anno_masks[0], dtype=np.uint8)

        if anno_masks.sum() == 0:
            continue

        pred_mask, pred_meta_info = retrieve_highest_iou(filename, preds_files, anno_masks)
        token_loc.append(float(pred_meta_info[3])/len(pred_meta_info[-1]) * 100)
        token_concept.append((pred_meta_info[0], pred_meta_info[-1]))

    nbins = 10
    bins = {i: 0 for i in range(nbins)}

    for loc in token_loc:
        for i in range(nbins):

            if loc >= i*10 and loc < (i+1)*10:
                bins[i] += 1

    np.save(os.path.join(args.when_out_dir, '%s_loc.npy'%args.meta_file.split('/')[-1].split('.')[0]), bins)
    np.save(os.path.join(args.when_out_dir, '%s_concept.npy'%args.meta_file.split('/')[-1].split('.')[0]), token_concept)

