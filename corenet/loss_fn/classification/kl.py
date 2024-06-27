#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.classification.base_classification_criteria import (
    BaseCriteria,
)


@LOSS_REGISTRY.register(name="kl", type="distribution")
class kl(BaseCriteria):
    
    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        # self.reduction = getattr(
        #     opts,
        #     "loss.classification.binary_cross_entropy.reduction",
        # )
        self.reduction = 'batchmean'

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != kl:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        # group.add_argument(
        #     "--loss.classification.binary-cross-entropy.reduction",
        #     type=str,
        #     default="mean",
        #     choices=["sum", "mean", "none", "batch_mean"],
        #     help="Specifies the reduction to apply to the output (default='mean')."
        #     " 'batch_mean' divides the sum of the loss only by the first dimension.",
        # )
        return parser
    def forward(
        self, prediction, *args, **kwargs
    ):
        return self._compute_loss(prediction)
    
    def _compute_loss(
        self, prediction, *args, **kwargs
    ) -> Tensor:

        stage3_tensor = prediction["stage3_tensor"]
        stage4_tensor = prediction["stage4_tensor"]
        div_by = 1.0
        if self.reduction == "batch_mean":
            div_by = stage3_tensor.shape[0]
        reduction = self.reduction if self.reduction != "batch_mean" else "sum"

        KLLoss = nn.KLDivLoss(reduction="batchmean")(
            stage3_tensor,
            stage4_tensor
        )
        return KLLoss / div_by

    def extra_repr(self) -> str:
        return f"\n\t reduction={self.reduction}"
