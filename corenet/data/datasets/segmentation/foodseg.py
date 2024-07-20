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


@DATASET_REGISTRY.register(name="foodseg", type="segmentation")
class FoodsegDataset(BaseImageSegmentationDataset):

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        split = "train" if self.is_training else "test"
        self.root = "/ML-A100/team/mm/models/FoodSeg103"
        ann_file = os.path.join(
            self.root, "ImageSets/{}.txt".format(split)
        )
        self.img_dir = os.path.join(self.root, "Images/img_dir/{}".format(split))
        self.ann_dir = os.path.join(self.root, "Images/ann_dir/{}".format(split))
        self.split = split
        with open(ann_file, 'r') as file:
            lines = file.readlines()
            self.ids = [line.strip() for line in lines]
        self.ignore_label = 255
        self.background_idx = 0

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int], *args, **kwargs
    ) -> Mapping[str, Union[Tensor, Mapping[str, Tensor]]]:
        """Returns the sample corresponding to the input sample index. Returned sample is transformed
        into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index)

        Returns:
            A dictionary with `samples` and `targets` as keys corresponding to input and labels of
            a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]["image"]: Shape is [Channels, Height, Width]
            output_data["targets"]["mask"]: Shape is [Height, Width]

        """
        crop_size_h, crop_size_w, img_index = sample_size_and_index

        _transform = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))
        path = self.ids[img_index]

        rgb_img = self.read_image_pil(os.path.join(self.img_dir, path))

        mask_file = os.path.join(self.ann_dir, path.replace('.jpg', '.png'))
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
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
