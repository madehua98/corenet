#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from functools import cached_property
from typing import Any, Dict, List, Tuple, Union

import torch
from pycocotools.coco import COCO

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
    BaseImageDataset,
)
from corenet.data.transforms.image_pil import BaseTransformation
import json


@DATASET_REGISTRY.register(name="food101", type="classification")
class food101_classification(BaseImageDataset):

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        self.root = getattr(opts, f"dataset.root")
        split = "train" if self.is_training else "test"
        self.split_file = os.path.join(self.root, 'meta_data', f"{split}_full.txt")
        self.image_class = {}
        with open(self.split_file, 'r') as file:
            for line in file:
                # 去除行尾的换行符，并按空格分割成两部分
                parts = line.strip().split()
                if len(parts) == 2:
                    file_path = parts[0]
                    value = int(parts[1])
                    # 将结果存储到字典中
                    file_path = os.path.join(self.root, "images", file_path)
                    self.image_class[file_path] = value
        self.images = list(self.image_class.keys())
        self.classes = list(set(self.image_class.values()))

        
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.images)

    def _training_transforms(
        self, size: Union[int, Tuple[int, int]]
    ) -> BaseTransformation:
        """Returns transformations applied to the input image in training mode.

        These transformations are the same as the 'BaseImageClassificationDataset'.

        Args:
            size: Size for resizing the input image. Expected to be an integer (width=height) or a tuple (height, width)

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        return BaseImageClassificationDataset._training_transforms(self, size)

    def _validation_transforms(
        self, *unused_args, **unused_kwargs
    ) -> BaseTransformation:
        """Returns transformations applied to the input in validation mode.

        These transformations are the same as the 'BaseImageClassificationDataset'.

        Returns:
            An instance of `corenet.data.transforms.image_pil.BaseTransformation.`
        """
        return BaseImageClassificationDataset._validation_transforms(self)

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index).

        Returns:
            A dictionary with `samples`, `sample_id` and `targets` as keys corresponding to input, index, and label of
            a sample, respectively.

        Shapes:
            The output data dictionary contains three keys (samples, sample_id, and target). The values of these
            keys has the following shapes:
                data["samples"]: Shape is [image_channels, image_height, image_width]
                data["sample_id"]: Shape is 1
                data["targets"]: Shape is [num_classes]
        """

        crop_size_h, crop_size_w, img_index = sample_size_and_index

        img_path = self.images[img_index]
        # target = torch.zeros(self.n_classes, dtype=torch.long)
        # target[self.image_class[img_path]] = 1
        target = self.image_class[img_path]
        input_img = self.read_image_pil(img_path)

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        data = transform_fn({"image": input_img})

        data["img_paths"] = img_path
        data["samples"] = data.pop("image")
        data["targets"] = target
        data["sample_id"] = img_index
        return data

    @cached_property
    def n_classes(self):
        return len(self.class_names)

    @cached_property
    def class_names(self) -> List[str]:
        """Returns the names of object classes in the food172 dataset."""
        
        return self.class_names