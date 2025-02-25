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
    parser.add_argument("--acc_file", type=str)
    parser.add_argument("--preds_dir", type=str)
    parser.add_argument("--meta_file", type=str, default="")
    parser.add_argument("--batchsize", default=8, type=int)
    parser.add_argument("--workers", default=0, type=int)
    return parser.parse_args()

def retrieve_auto_select(filename, preds_files, meta_info):
    for row in meta_info:
        fname = str(int(row["question_id"])+1)+'.jpg'
        if fname == filename.split('/')[-1]:
            selectedf = row["selected file"]
            pred_mask = cv2.imread(selectedf, 0)
            pred_mask[pred_mask==255] = 1
            return pred_mask

    return None

def load_meta_json_info(answer_file):
    datas = []
    with open(answer_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            datas.append(data)
    return datas


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

    acc_df = pd.read_csv(args.acc_file)

    # 0: grounding only failes, 1: VQA only fails, 2: both fails, 3: both succeeds
    histogram = {x: 0 for x in range(4)}

    auto_select = False
    if 'llava-1.5-7b' in args.acc_file or 'llava-1.5-13b' in args.acc_file or 'cambrian-8b' in args.acc_file:
        auto_select = True
        assert args.meta_file != "", "Meta file is missing"
        meta_info = load_meta_json_info(args.meta_file)

    prev_acc = False
    prev_success_iou = False

    for idx, minibatch in tqdm(enumerate(dataloader)):
        for i in range(len(minibatch)):

            filename = minibatch[i]['file_name']
            index = int(filename.split('/')[-1].split('.')[0]) - 1

            anno_masks = minibatch[i]['masks']
            if type(anno_masks) == list:
                assert len(anno_masks) == 1, "More than one mask Issue"
                anno_masks = np.array(anno_masks[0], dtype=np.uint8)

            if auto_select:
                pred_mask = retrieve_auto_select(filename, preds_files, meta_info)
            else:
                pred_mask = cv2.imread(preds_files[idx * args.batchsize + i], 0)
                pred_mask[pred_mask==255] = 1

            success_iou = 0
            if anno_masks.sum() != 0:
                iou = compute_iou(pred_mask, anno_masks)
                success_iou = (iou > 0.5)
            else:
                shape = anno_masks.shape[-2:]
                area = pred_mask.sum() / (shape[0] * shape[1])
                success_iou = (area < 0.5)

            accuracy = (acc_df.iloc[index, 1] == 1)

            if index%2 == 0:
                prev_accuracy = accuracy
                prev_success_iou = success_iou
            else:
                accuracy = accuracy and prev_accuracy
                success_iou = success_iou and prev_success_iou

                if accuracy and success_iou:
                    histogram[3] += 1
                elif not accuracy and not success_iou:
                    histogram[2] += 1
                elif accuracy and not success_iou:
                    histogram[1] += 1
                else:
                    histogram[0] += 1

                prev_accuracy = False
                prev_success_iou = False

    assert np.array([v for v in histogram.values()]).sum() == 150, "Missing Images in Frequency count"
    print("Histogram = ", histogram)
