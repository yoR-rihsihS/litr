import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

def scale_crop(image, bboxes, size):
    """
    Scale image to fixed width and crop vertically around the objects.
    Args:
        - image: np.ndarray (H, W, C)
        - bboxes: np.ndarray (N, 4) in [cx, cy, w, h] format
        - size: Tuple[int, int] - desired output size (h0, w0)
    Returns:
        - image: Cropped and scaled image
        - bboxes: Updated bboxes in [cx, cy, w, h] format
    """
    h0, w0 = size
    h, w, _ = image.shape
    factor = w0 * 1.0 / w

    # Resize image to fixed width
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    bboxes = bboxes * factor

    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1, y1, x2, y2 = (cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2)
    bboxes = np.stack([x1, y1, x2, y2], axis=1)

    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_u = max_bbox[1]
    max_d = h - max_bbox[3]

    dis = min(max_u, max_d)
    dis = min(dis, h-h0)
    crop = int(random.uniform(0, dis))
    if max_u < max_d:
        x0 = crop
    else:
        x0 = h - h0 - crop 

    x0 = min(x0, h-h0)
    x0 = max(0, x0)

    image = image[x0:x0+h0, :, :]
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - x0

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
    bboxes = np.stack([cx, cy, w, h], axis=1)

    return image, bboxes

def random_scale(image, bboxes):
    """
    Randomly scale image and annotations, then pad to original size.
    Args:
        - image (Tensor): (C, H, W)
        - bboxes (Tensor): (N, 4) in [cx, cy, w, h]
    Returns:
        - image (Tensor): scaled and padded image
        - bboxes (Tensor): scaled bboxes
    """
    C, H, W = image.shape
    factor = random.uniform(0.5, 0.9)
    new_H, new_W = int(H * factor), int(W * factor)

    image_resized = F.interpolate(image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
    bboxes = bboxes * factor

    pad_bottom = H - new_H
    pad_right = W - new_W
    padding = [0, pad_right, 0, pad_bottom]  # left, right, top, bottom
    image_padded = F.pad(image_resized, padding, mode='constant', value=0)

    return image_padded, bboxes

def random_crop(image, bboxes, min_crop_size=480):
    """
    Randomly crops an area around the objects and resizes the crop back to the original size.
    Args:
        - image: Tensor (C, H, W)
        - bboxes: Tensor (N, 4) in [cx, cy, w, h]
        - min_crop_size: minimum crop height/width allowed
    Returns:
        - image_resized: Tensor (C, H, W) - cropped and resized back to original size
        - bboxes: Tensor (N, 4) - adjusted to resized image
    """
    C, H, W = image.shape

    bboxes = box_cxcywh_to_xyxy(bboxes)  # Convert to [x1, y1, x2, y2]

    # Compute tightest bbox enclosing all objects
    xy_min = torch.min(bboxes[:, 0:2], dim=0).values
    xy_max = torch.max(bboxes[:, 2:4], dim=0).values
    max_bbox = torch.cat([xy_min, xy_max], dim=0)  # [x1, y1, x2, y2]

    # Calculate possible crop limits
    max_l_trans = max_bbox[0].item()
    max_u_trans = max_bbox[1].item()
    max_r_trans = W - max_bbox[2].item()
    max_d_trans = H - max_bbox[3].item()

    # Random crop boundaries
    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = min(W, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = min(H, int(max_bbox[3] + random.uniform(0, max_d_trans)))

    crop_w = crop_xmax - crop_xmin
    crop_h = crop_ymax - crop_ymin

    if crop_w < min_crop_size or crop_h < min_crop_size:
        return image, box_xyxy_to_cxcywh(bboxes)  # Skip crop if too small

    # Crop the image
    cropped_image = image[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Resize crop back to original size
    resized_image = F.interpolate(cropped_image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)

    # Compute scale factors
    scale_x = W / crop_w
    scale_y = H / crop_h

    # Adjust bboxes
    bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - crop_xmin) * scale_x
    bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - crop_ymin) * scale_y

    return resized_image, box_xyxy_to_cxcywh(bboxes)

def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    Randomly distort image color. Adjust brightness, hue, saturation, value.
    Args:
        - img: a BGR uint8 format OpenCV image. HWC format.
    '''
    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img