import glob
import argparse
import numpy as np
from pixcvbench.dataset.register_pixcvbench import register_new_dataset
from pixcvbench.dataset.custom_coco_dataset import CustomCOCODataset
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
    parser = argparse.ArgumentParser(description="PixCVBench")
    parser.add_argument("--preds_dir", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    roots = [os.path.join(args.dataset_root, 'ADE20K'), os.path.join(args.dataset_root, 'COCO')]
    register_new_dataset(roots)

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    dataset = CustomCOCODataset(mean, std)

    # Dataloading
    dataloader = torchdata.DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                                      drop_last=False, num_workers=args.workers,
                                      collate_fn=trivial_batch_collator)

    miou = 0
    nimages = 0

    for idx, minibatch in enumerate(tqdm(dataloader)):
        for i in range(len(minibatch)):

            filename = minibatch[i]['file_name']
            anno_masks = minibatch[i]['masks']
            if type(anno_masks) == list:
                final_anno_masks = anno_masks[0]
                for mask in anno_masks:
                    final_anno_masks[mask==1] = 1
                anno_masks = final_anno_masks
            anno_masks = np.array(anno_masks, np.uint8)

            filename = filename.split('/')[-1].split('.')[0] + '.png'
            if not os.path.exists(os.path.join(args.preds_dir, filename)):
                filename = filename.replace('png', 'jpg')
            pred_mask = cv2.imread(os.path.join(args.preds_dir, filename), 0)
            pred_mask[pred_mask==255] = 1

            pred_mask = np.array(cv2.resize(np.array(pred_mask, np.float32), anno_masks.shape[::-1]), np.uint8)
            iou = compute_iou(pred_mask, anno_masks)

            miou += iou
            nimages += 1

    print("Total mIoU = ", float(miou / nimages))

