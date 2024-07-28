import argparse
import os
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import cv2
from torch import Tensor

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.segmentation.base_segmentation import (
    BaseImageSegmentationDataset,
)
from PIL import Image

@DATASET_REGISTRY.register(name="uec", type="segmentation")
class FoodsegDataset(BaseImageSegmentationDataset):

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        split = "train" if self.is_training else "test"
        self.root = "/ML-A100/team/mm/models/UECFOODPIXCOMPLETE/data"
        if split == 'train':
            ann_file = os.path.join(
                self.root, "train9000.txt"
                )
        elif split == 'test':
            ann_file = os.path.join(
                self.root, "test1000.txt"
                )
        self.img_dir = os.path.join(self.root, "UECFoodPIXCOMPLETE/{}/img".format(split))
        self.ann_dir = os.path.join(self.root, "UECFoodPIXCOMPLETE/{}/mask".format(split))
        self.split = split
        with open(ann_file, 'r') as file:
            lines = file.readlines()
            self.ids = [line.strip() + '.jpg' for line in lines]
        self.ignore_label = 255
        self.background_idx = 0
    def read_image_pil(self, image_path):
        # Read the image using cv2
        image = cv2.imread(image_path)
        # Convert the image from BGR (OpenCV format) to RGB (PIL format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image from NumPy array to PIL image
        pil_image = Image.fromarray(image)
        return pil_image


    def read_mask_pil(self, mask_path):
        # Read the mask using cv2
        mask = cv2.imread(mask_path)
        red_channel = mask[:, :, 2] 
        # Convert the mask from NumPy array to PIL image
        pil_mask = Image.fromarray(red_channel)
        return pil_mask

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int], *args, **kwargs
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:

        crop_size_h, crop_size_w, img_index = sample_size_and_index

        _transform = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))
        path = self.ids[img_index]

        rgb_img = self.read_image_pil(os.path.join(self.img_dir, path))

        mask_file = os.path.join(self.ann_dir, path.replace('.jpg', '.png'))
        mask = self.read_mask_pil(mask_file)
        data = {"image": rgb_img, "mask": None if self.is_evaluation else mask}
        data = _transform(data)

        if self.is_evaluation:
            # for evaluation purposes, resize only the input and not mask
            data["mask"] = mask

        output_data = {"samples": data["image"], "targets": data["mask"]}

        if self.is_evaluation:
            im_width, im_height = rgb_img.size
            img_name = path.replace("jpg", "png")
            mask = output_data.pop("targets")
            output_data["targets"] = {
                "mask": mask,
                "file_name": img_name,
                "im_width": im_width,
                "im_height": im_height,
            }

        return output_data

    def __len__(self) -> int:
        return len(self.ids)
