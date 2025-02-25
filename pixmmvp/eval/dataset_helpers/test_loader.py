import numpy as np
from register_mmvp import register_new_dataset
from custom_coco_dataset import CustomCOCODataset
import torch.utils.data as torchdata
from tqdm import tqdm as tqdm
from detectron2.data.build import trivial_batch_collator
import cv2
import os
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
    batch_size = 8
    num_workers = 0
    root = '/home/ivulab/Code/MMVP/MMVP/'
    register_new_dataset(root)

    out_dir = '/home/ivulab/Code/MMVP/MMVP/MMVP_Seg/'
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    dataset = CustomCOCODataset(mean, std)

    # Dataloading
    dataloader = torchdata.DataLoader(dataset, batch_size=batch_size, #sampler=sampler,
                                      drop_last=False, num_workers=num_workers,
                                      collate_fn=trivial_batch_collator)

    for minibatch in tqdm(dataloader):
        for i in range(len(minibatch)):

            filename = minibatch[i]['file_name']
            img = minibatch[i]['image']
            anno_masks = minibatch[i]['masks']

            img = np.array(img.permute(1,2,0))
            img = denormalize(img, mean, std)
            viz_img = visualize_masks(img, anno_masks)

            if not os.path.exists(os.path.join(out_dir, 'images')):
                os.makedirs(os.path.join(out_dir, 'images'))

            cv2.imwrite(os.path.join(out_dir, 'images/', filename.split('/')[-1]), viz_img)
