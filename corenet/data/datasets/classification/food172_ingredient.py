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


@DATASET_REGISTRY.register(name="food172_ingredient", type="classification")
class food172ingredient_lassification(BaseImageDataset):
    """`COCO <https://cocodataset.org/#home>`_ dataset for multi-label object classification.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        split = "train" if self.is_training else "text"
        ann_file = os.path.join(
            self.root, "SplitAndIngreLabel/{}_IngreLabel.json".format(split)
        )
        self.image_ingredient = []
        with open(ann_file, mode='r') as file:
            lines = file.readlines()
            for line in lines:
                line = json.loads(line)
                self.image_ingredient.append(line)
                
        self.img_dir = os.path.join(self.root, "ready_chinese_food")
        self.images = list(self.image_ingredient.keys())
        self.class_names_filename = os.path.join(
            self.root, "SplitAndIngreLabel/IngredientList.txt"
        )
        with open(self.class_names_filename, mode='r') as file:
            lines = file.readlines()
            self.class_names = [line.strip() for line in lines]
        
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
        target = torch.zeros(self.n_classes, dtype=torch.long)
        for ingredient in self.image_ingredient[img_path]:
            target[ingredient] = 1

        img_path = os.path.join(self.img_dir, img_path)
        input_img = self.read_image_pil(img_path)

        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        data = transform_fn({"image": input_img})

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