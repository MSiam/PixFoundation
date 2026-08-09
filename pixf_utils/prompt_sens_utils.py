import numpy as np
import random
import string
import cv2

def alter_text_util(text, N, error_type):
    rnd_indices = np.random.randint(2, len(text)-2, N)

    if error_type == 1:
        # Insertion
        for idx in rnd_indices:
            char = random.choice(string.ascii_letters)
            text = text[:idx] + char + text[idx:]
    elif error_type == 2:
        # Ommission
        for idx in rnd_indices:
            text = text[:idx] + text[idx+1:]
    else:
        # Transpositions
        for idx in rnd_indices:
            temp = text[idx]
            if text[idx+1] != ' ':
                text = text[:idx] + text[idx+1] + temp + text[idx+2:]
            else:
                text = text[:idx-1] + temp + text[idx-1] + text[idx+1:]
    return text

def guided_crop(img, mask, hidden_percent=-1.0):
    if mask.sum() == 0:
        return img, mask

    h, w = img.shape[:2]
    counter = 0
    while True:
        counter += 1
        if counter > 1000:
            return img, mask

        if hidden_percent == -1.0:
            hidden_percent = 0.5 + np.random.rand() * 0.3 # generate between 30 - 60%

        bb = list(cv2.boundingRect(mask))

        tx = bb[0]
        ty = bb[1]

        bbh_h, bbh_w = int(bb[3]*hidden_percent), int(bb[2]*hidden_percent)

        # 30 pixels is used specifically for Qwen type of models haspatch size 14 and reduces size by 2 so at 28 pixels min.
        if bbh_h < 30:
            bbh_h = 30
        if bbh_w < 30:
            bbh_w = 30

        bbh = [tx, ty] + [bbh_w, bbh_h]
        if mask[bbh[1]:bbh[1]+bbh[3], bbh[0]:bbh[0]+bbh[2]].sum() > 0.2 * mask.sum():
            break

    mask = mask[:bbh[1]+bbh[3], :bbh[0]+bbh[2]]
    img = img[:bbh[1]+bbh[3], :bbh[0]+bbh[2]]

    return img, mask

def guided_maskout(img, mask, hidden_percent=-1.0):
    if mask.sum() == 0:
        return img, mask

    h, w = img.shape[:2]
    counter = 0
    while True:
        counter += 1
        if counter > 1000:
            return img, mask

        bb = list(cv2.boundingRect(mask))
        if hidden_percent == -1.0:
            hidden_percent = np.random.rand() * 0.8 # generate between 0 - 50%
            tx = np.random.randint(bb[0], bb[0] + bb[2])
            ty = np.random.randint(bb[1], bb[1] + bb[3])
        else:
            tx, ty = bb[:2]

        bbh_h, bbh_w = int(bb[3]*hidden_percent), int(bb[2]*hidden_percent)
        bbh = [tx, ty] + [bbh_w, bbh_h]

        if mask[bbh[1]:bbh[1]+bbh[3], bbh[0]:bbh[0]+bbh[2]].sum() > 0.2 * mask.sum():
            break

    mask[bbh[1]:bbh[1]+bbh[3], bbh[0]:bbh[0]+bbh[2]] = 0
    img[bbh[1]:bbh[1]+bbh[3], bbh[0]:bbh[0]+bbh[2]] = (0,0,0)

    return img, mask
