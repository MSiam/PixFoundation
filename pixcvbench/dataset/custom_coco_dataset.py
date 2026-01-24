from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import itertools
import torchvision.ops as ops
import torch
import matplotlib.pyplot as plt
import cv2
import pycocotools.mask as mask_util

from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import detection_utils as utils
from detectron2.data.catalog import DatasetCatalog
from panopticapi.utils import rgb2id

class CustomCOCODataset(Dataset):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = True

        self.dataset_categories = {}
        self.dataset_dicts = {}

        dataset_dict = DatasetCatalog.get('cvbench_coco')
        self.dataset_dicts['coco'] = dict(itertools.chain.from_iterable([dataset_dict[0]]))
        self.dataset_categories['coco'] = dataset_dict[1]

        dataset_dict = DatasetCatalog.get('cvbench_ade20k')
        self.dataset_dicts['ade20k'] = dict(itertools.chain.from_iterable([dataset_dict[0]]))
        self.dataset_categories['ade20k'] = dataset_dict[1]

        self.dataset_img_ids = ['ade20k_%d'%k for k in self.dataset_dicts['ade20k'].keys()]
        self.dataset_img_ids += ['coco_%d'%k for k in self.dataset_dicts['coco'].keys()]

    def __len__(self):
        return len(self.dataset_img_ids)

    def _annToRLE(self, segm, h, w):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    def _annToMask(self, ann, width, height):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self._annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def _convertCategory(self, cat_id, dataset_name):
        category_infos = self.dataset_categories[dataset_name]
        for info in category_infos:
            if info["id"] == cat_id:
                return info["name"]
        return None

    def __getitem__(self, idx):
        img_id = self.dataset_img_ids[idx]
        if 'coco' in img_id:
            dataset_name = 'coco'
        elif 'ade20k' in img_id:
            dataset_name = 'ade20k'

        img_id = int(img_id.split('_')[1])

        infos = self.dataset_dicts[dataset_name][img_id]
        filename = infos[0]['file_name']

        img = Image.open(filename)
        w, h = img.size

        masks = []
        classes = []
        class_names = []

        for info in infos:
            if info["segments_info"]["segments"] is None:
                mask = np.zeros((h, w))
            else:
                mask_w, mask_h = info["segments_info"]["size"]
                mask = self._annToMask(info["segments_info"]["segments"], mask_w, mask_h)
                if mask.shape[0] != h or mask.shape[1] != w:
                    mask = cv2.resize(mask, (w, h))
            masks.append(mask)

            classes.append(info["segments_info"]["category"])
            class_names.append(self._convertCategory(info["segments_info"]["category"], dataset_name))

        final = {'masks': masks, 'classes': classes, 'class_names': class_names, 'dataset_name': dataset_name,
                 'image_id': img_id}

        # normalization preproc
        if self.normalize:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=self.mean, std=self.std)])
            if img.mode == 'L':
                img = Image.fromarray(np.stack([np.array(img)]*3).transpose(1,2,0))
            img = transform(img)

        final['image'] = img
        final['file_name'] = filename
        return final
