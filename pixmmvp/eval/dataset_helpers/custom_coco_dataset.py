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
        dataset_dicts = [DatasetCatalog.get('mmvp_val')]
        self.dataset_dicts = dict(itertools.chain.from_iterable(dataset_dicts))
        self.dataset_img_ids = list(self.dataset_dicts.keys())

    def __len__(self):
        return len(self.dataset_dicts)

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

    def __getitem__(self, idx):
        img_id = self.dataset_img_ids[idx]
        infos = self.dataset_dicts[img_id]
        filename = infos[0]['file_name']
        img = Image.open(filename)
        w, h = img.size

        # This will always be one mask as we dont have instances, it is always single object in PixMMVP
        masks = []

        for info in infos:
            if info["segments_info"]["segments"] is None:
                final_mask = np.zeros((w, h))
            else:
                final_mask = None
                for ann in info["segments_info"]["segments"]:
                    mask = self._annToMask(ann, w, h)
                    if final_mask is None:
                        final_mask = mask
                    else:
                        final_mask[mask == 1] = 1
            masks.append(final_mask)

        final = {'masks': masks}

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
