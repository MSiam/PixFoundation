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
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

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

    for idx, minibatch in tqdm(enumerate(dataloader)):
        for i in range(len(minibatch)):

            filename = minibatch[i]['file_name']
            anno_masks = minibatch[i]['masks']
            if type(anno_masks) == list:
                assert len(anno_masks) == 1, "More than one mask Issue"
                anno_masks = np.array(anno_masks[0], dtype=np.uint8)

            pred_mask = cv2.imread(preds_files[idx * args.batchsize + i], 0)
            pred_mask[pred_mask==255] = 1

            iou = compute_iou(pred_mask, anno_masks)

            miou += iou
            nimages += 1

    print("Total mIoU = ", float(miou / nimages))

