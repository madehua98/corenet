#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List

from corenet.data.datasets.classification.imagenet_r import (
    IMAGENET_R_CLASS_SUBLIST,
    ImageNetRDataset,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification import (
    ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.base_zero_shot_image_classification import (
    BaseZeroShotImageClassificationDataset,
)
from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.imagenet import (
    IMAGENET_CLASS_NAMES,
)


@ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY.register(name="imagenet_r")
class ImageNetRDatasetZeroShot(
    BaseZeroShotImageClassificationDataset, ImageNetRDataset
):
    """ImageNet-R dataset for zero-shot evaluation of Image-text models."""

    @property
    def class_names(self) -> List[str]:
        """Return the name of the classes present in the dataset."""
        return [
            IMAGENET_CLASS_NAMES[ImageNetRDataset.class_id_to_imagenet_class_id(i)]
            for i in range(len(IMAGENET_R_CLASS_SUBLIST))
        ]
