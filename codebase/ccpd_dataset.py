import re
import cv2
import numpy as np
import pandas as pd
from .generate_splits import generate_splits

import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

from .data_aug import random_color_distort, scale_crop, random_scale, random_crop

class CCPDDataset(Dataset):
    def __init__(self, path, eval_size, normalize_mean, normalize_std, mode="train"):
        generate_splits()
        assert mode in ["train", "eval"], "mode should be either 'train' or 'eval'."
        self.eval_size = eval_size
        self.mode = mode
        self.affine = v2.RandomAffine(degrees=30, shear=[-10, 10, -10, 10], interpolation=InterpolationMode.BILINEAR)
        self.blur = v2.GaussianBlur(kernel_size=31, sigma=(0.1, 5.0))
        if normalize_mean is not None and normalize_std is not None:
            self.normalize = v2.Normalize(mean=normalize_mean, std=normalize_std)
        else:
            self.normalize = None
        if mode == "train":
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        elif mode == "eval":
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.eval_size, interpolation=InterpolationMode.BILINEAR),
                v2.ToDtype(torch.float32, scale=True),
            ])
        self.data_frame = pd.read_csv(path, header=None, names=["file_path"])

    def __len__(self):
        return len(self.data_frame)
    
    def _get_annotations(self, image_path):
        name = image_path[image_path.find('/') + 1 : -4]
        attributes = name.split('-')
        bbox = attributes[2]
        labels = attributes[4]
        bbox = re.findall(r'\d+', bbox)
        bboxes = [int(x) for x in bbox]
        bboxes = [[(bboxes[2]+bboxes[0])/2, (bboxes[3]+bboxes[1])/2, (bboxes[2]-bboxes[0]), (bboxes[3]-bboxes[1])]]
        labels = re.findall(r'\d+', labels)
        labels = [[int(i) for i in labels]]
        return np.array(bboxes), np.array(labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.data_frame['file_path'].iloc[idx]
        bboxes, labels = self._get_annotations(image_path)

        img = cv2.imdecode(np.fromfile('../CCPD2019/' + image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.mode == 'train':
            if np.random.randint(2):
                img = random_color_distort(img)
            img, bboxes = scale_crop(img, bboxes, self.eval_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="CXCYWH", canvas_size=(img.shape[0], img.shape[1])) # Take care of canvas size
        img, bboxes = self.transform((img, bboxes))

        if self.mode == 'train':
            flag = np.random.randint(6) # 0, 1, 2, 3, 4, 5
            if flag < 2:
                bboxes = bboxes.as_subclass(torch.Tensor)
                img, bboxes = random_scale(img, bboxes) # zoom out
            elif flag == 2:
                bboxes = bboxes.as_subclass(torch.Tensor)
                img, bboxes = random_crop(img, bboxes) # zoom in
            elif flag < 5:
                img, bboxes = self.affine((img, bboxes))  # rotate and shear
                bboxes = bboxes.as_subclass(torch.Tensor)
            elif flag == 5:
                img, bboxes = self.blur((img, bboxes)) # blur
                bboxes = bboxes.as_subclass(torch.Tensor)

        if self.normalize is not None:
            img = self.normalize(img)
        bboxes = bboxes / self.eval_size[0] # [num_objects, 4]
        labels = torch.tensor(labels, dtype=torch.long) # [num_objects, 7]

        target = {
            'label_0' : labels[:, 0], # [num_objects]
            'label_1' : labels[:, 1], # [num_objects]
            'label_2' : labels[:, 2], # [num_objects]
            'label_3' : labels[:, 3], # [num_objects]
            'label_4' : labels[:, 4], # [num_objects]
            'label_5' : labels[:, 5], # [num_objects]
            'label_6' : labels[:, 6], # [num_objects]
            'boxes': bboxes.to(torch.float32), # [num_objects, 4]
        }

        return img, target