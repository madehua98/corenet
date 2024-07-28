from transformers import CLIPModel
import torch.nn as nn
from typing import Callable, Dict, Optional
from functools import partial

from functools import partial
from typing import List, Tuple
from dataclasses import dataclass, replace
from typing import Callable, Optional, Union, Tuple, List, Sequence
import math, time
from torch.jit import Final
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.checkpoint import checkpoint
from timm.models.layers import create_attn, get_norm_layer, get_norm_act_layer, create_conv2d, make_divisible, trunc_normal_tf_
from dataclasses import dataclass, field

from timm.layers import to_2tuple, DropPath, Format # , trunc_normal_
from timm.layers.norm_act import _create_act
from timm.models._registry import register_model
from timm.models._manipulate import named_apply, checkpoint_seq
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import get_act_layer, Type, LayerType, Mlp, Block, PatchEmbed, VisionTransformer, checkpoint_filter_fn, get_init_weights_vit, init_weights_vit_timm, _load_weights 
import logging
from collections import OrderedDict
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from corenet.modeling.layers import (
    ConvLayer2d,
    Dropout,
    Identity,
    LinearLayer,
    MaxPool2d,
    PositionalEmbedding,
    TransposeConvLayer2d,
    get_normalization_layer,
)
from corenet.modeling.misc.common import parameter_list
from corenet.modeling.misc.init_utils import initialize_conv_layer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
#from corenet.modeling.models.classification.config.open_vit_config import get_configuration
from corenet.modeling.modules import FlashTransformerEncoder, TransformerEncoder
from corenet.utils import logger

import torch
import torch.nn as nn
from typing import List, Tuple

import math

@MODEL_REGISTRY.register(name="openvit", type="classification")
class OpenClipViT(BaseImageEncoder):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        
        # 加载 CLIP 模型并提取 ViT 部分
        clip_model = CLIPModel.from_pretrained('/ML-A100/team/mm/models/vit-base')
        self.model = clip_model.vision_model
        
        self.num_classes = getattr(self.opts, "model.classification.n_classes")
        print(self.num_classes)
        self.classifier = nn.Linear(self.model.config.hidden_size, self.num_classes)
        self.model_conf_dict = {
            "conv1": {"in": 3, "out": self.model.config.hidden_size},
            "layer1": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "layer2": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "layer3": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "layer4": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "layer5": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "exp_before_cls": {"in": self.model.config.hidden_size, "out": self.model.config.hidden_size},
            "cls": {"in": self.model.config.hidden_size, "out": self.n_classes},
        }
        # if self.num_classes == 104 or self.num_classes == 102:
        #     self.model.embeddings.position_embedding =  nn.Embedding(1025,self.model.config.hidden_size)
        #     self.model.embeddings.position_ids = torch.tensor([i for i in range(self.model.embeddings.position_embedding.num_embeddings)])
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == VisionTransformer:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--model.classification.openvit.mode",
                type=str,
                default="base",
                choices=["tiny", "small", "base", "large", "huge"],
                help="ViT mode. Default is base.",
            )
            group.add_argument(
                "--model.classification.openvit.dropout",
                type=float,
                default=0.0,
                help="Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.openvit.stochastic-dropout",
                type=float,
                default=0.0,
                help="Stochastic Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.openvit.norm-layer",
                type=str,
                default="layer_norm",
                help="Normalization layer to be used in Transformer layer. Defaults to LayerNorm.",
            )

            group.add_argument(
                "--model.classification.openvit.sinusoidal-pos-emb",
                action="store_true",
                default=False,
                help="Use sinusoidal instead of learnable positional embedding. Defaults to False.",
            )
            group.add_argument(
                "--model.classification.openvit.no-cls-token",
                action="store_true",
                default=False,
                help="Do not use classification token. Defaults to False.",
            )

            group.add_argument(
                "--model.classification.openvit.use-simple-fpn",
                action="store_true",
                default=False,
                help="Add simple FPN for down-stream tasks (e.g., detection). Defaults to False.",
            )
            group.add_argument(
                "--model.classification.openvit.use-flash-attention",
                action="store_true",
                default=False,
                help="Use Transformer layers with flash attention for efficiently computing scaled dot-product attention. Defauls to False.",
            )

        return parser

    # def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.classifier(x)
    #     return x

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.forward_features(x)
    #     x = self.forward_classifier(x)
    #     return x
    

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Extract patch embeddings from input image tensor.

        Args:
            x: Input image tensor of size [batch, image channels, height, width]

        Returns:
            A tensor containing patch embeddings. The size of the tensor is [batch, number of patches, embedding dim].
        """
        # input is of shape [batch, image channels, height, width]. image channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]  # 256

        # [batch, image channels, height, width] --> [batch, embedding dim, number of patches along height, number of patches along width]
        patch_emb = self.patch_emb(x)  # [256,768,14,14]
        num_patches_height, num_patches_width = patch_emb.shape[-2:]  # 14  14

        # [batch, embedding dim, number of patches along height, number of patches along width] --> [batch, embedding dim, number of patches]
        patch_emb = patch_emb.flatten(2)  # [256,768,196]
        # [batch, embedding dim, number of patches] --> [batch, number of patches, embedding dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()  # [256,196,768]

        num_patches = patch_emb.shape[1]  # 196
        # we resize the positional encodings dynamically.
        pos_emb = self.pos_embed(num_patches).to(patch_emb.dtype)  # [1,196,768]

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, embedding dim] --> [batch, 1, embedding dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [256,1,768]
            # Concat([batch, 1, embedding dim], [batch, number of patches, embedding dim]) --> [batch, number of patches + 1, embedding dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)  # [256,197,768]

        # dropout
        patch_emb = self.emb_dropout(patch_emb)  # [256,197,768]
        return patch_emb, (num_patches_height, num_patches_width)

    def _features_from_transformer(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Helper function to extract patch embeddings and learn inter-patch representations using transformers.

        Args:
            x: Input image tensor of size [batch, image channels, Height, Width]

        Returns:
            A tensor containing contextualized patch embeddings.The size of the tensor is [batch, number of patches, embedding dimension]. It also
            returns a tuple containing the number of patches along height and width dimensions.
        """
        # x, (n_h, n_w) = self.extract_patch_embeddings(x)  # [256,3,224,224]->[256,197,768]
        # x = self.transformer(x)  # [256,197,768]
        x = self.model(x)[0]
        h_w = self.model.embeddings.position_embedding.num_embeddings
        h = w = math.sqrt(int(h_w))
        return x, h, w

    def extract_features(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Helper function for extraction features.

        Args:
            x: Input image tensor of size [batch, image channels, height, width].
            return_image_embeddings: When enabled, image embeddings are also returned.

        Returns:
            If 'return_image_embeddings=True', then both CLS_TOKEN and image embeddings are returned. Otherwise,
            CLS_TOKEN embedding and None are returned.

            The shape of CLS_TOKEN embedding is [batch, embedding dim] while the shape of image embeddings is
            [batch, embedding dim, num. patches height, num. patches width].
        """

        # [Batch, image channels, height, Width] --> [batch, CLS_TOKEN + number of patches, embedding dim]
        x, n_h, n_w = self._features_from_transformer(x)  # [256,197,768]

        # [batch, CLS_TOKEN + num. patches, embedding dim] --> [batch, embedding dim], [batch, number of patches, embedding dim]
        cls_embedding, image_embedding = torch.split(
            x, split_size_or_sections=[1, x.shape[1] - 1], dim=1
        )  # [256,1,768]  [256,196,768]
        cls_embedding = cls_embedding.squeeze(1)  # [256,768]

        if return_image_embeddings:
            # reshape image embedding to 4-D tensor
            # [batch, number of patches, embedding dim] --> [batch, embedding dim, number of patches]
            image_embedding = image_embedding.transpose(1, 2).contiguous()
            # [batch, embedding dim, number of patches] --> [batch, embedding dim, number of patches along height, number of patches along width]
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], -1, int(n_h), int(n_w)
            )

            return cls_embedding, image_embedding
        else:
            return cls_embedding, None

    def forward_classifier(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward function for classification tasks.

        Args:
            x: Input image tensor of size [batch, image channels, height, width].
            return_image_embeddings: When enabled, image embeddings are also returned.

        Returns:
            The logits computed for CLS token are returned. If kwargs contain 'return_image_embeddings', then image embeddings
            are also returned.

            The shape of logits is [batch, number of classes] while the shape of image embeddings is
            [batch, embedding dim, num. patches height, num. patches width].
        """
        cls_embedding, image_embedding = self.extract_features(
            x, return_image_embeddings
        )  # [256,768] None
        # classify based on CLS token
        logits = self.classifier(cls_embedding)  # [256,24320]
        return logits, image_embedding

    def forward(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward fucntion for ViT.

        Args:
            x: Input image tensor of shape [Batch, 3, Height, Width].
            return_image_embeddings: When enabled, image embeddings are also returned.
 
        Returns:
            The output of ViT model can be one of the following:
            1. If range augmentation is enabled, then a dictionary is returned with following keys
                'augmented_tensor': Contains the output after applying RangeAugment.
                'logits': Logit tensor
                'image_embeddings': Optionally tensor containing image embeddings
            2. If range augment is not enabled and return_image_embeddings is enabled, then a
               dictionary is returned with 'logits' and 'image_embeddings' keys.
            3. A logit tensor is returned.
        """
        # forward 计算forward_classifier(计算extract_features(计算_features_from_transformer(计算extract_patch_embeddings(计算patch_emb))))，返回logits
        if return_image_embeddings or self.neural_augmentor is not None:
            out_dict = {"augmented_tensor": None}
            if self.training and self.neural_augmentor is not None:
                # neural augmentor is applied during training  only
                x = self.neural_augmentor(x)
                out_dict.update({"augmented_tensor": x})
            logits, image_embedding = self.forward_classifier(
                x, return_image_embeddings
            )  # [256,24320]
            out_dict.update({"logits": logits})
            if image_embedding is not None:
                out_dict.update({"image_embeddings": image_embedding})
            return out_dict
        else:
            logits, _ = self.forward_classifier(x)
            return logits

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
    ) -> Dict[str, Tensor]:

        out_dict = {}
        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        cls_emb, x = self.extract_features(x, return_image_embeddings=True)
        out_dict["cls_embedding"] = cls_emb
        out_dict["out_l1"] = None
        out_dict["out_l2"] = None
        out_dict["out_l3"] = None
        out_dict["out_l4"] = None
        if use_l5_exp:
            out_dict["out_l5"] = None
            out_dict["out_l5_exp"] = x
        else:
            out_dict["out_l5"] = x
            out_dict["out_l5_exp"] = None
        return out_dict

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpoint module class.

        For ViT, the activation checkpoint module class is TransformerEncoder or FlashTransformerEncoder.
        """
        return FlashTransformerEncoder if self.use_flash_attn else TransformerEncoder

    def get_fsdp_wrap_policy(
        self,
    ) -> Optional[Callable[[torch.nn.Module, bool, int], bool]]:
        """Returns the FSDP wrapping policy.

        For ViT, we use the Transfomer's wrapping policy.
        """
        vit_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                FlashTransformerEncoder if self.use_flash_attn else TransformerEncoder
            },
        )
        return vit_auto_wrap_policy

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpoint module class.

        For ViT, the activation checkpoint module class is TransformerEncoder or FlashTransformerEncoder.
        """
        return FlashTransformerEncoder

# from transformers import CLIPModel
# from transformers.models.clip.modeling_clip import CLIPVisionTransformer  # 导入视觉模型



# # 定义一个类来继承你的模型和BaseImageEncoder
# @MODEL_REGISTRY.register(name="openvit", type="classification")
# class OpenViTModel(BaseImageEncoder, CLIPVisionTransformer):
#     # 定义类变量
#     clip_model = CLIPModel.from_pretrained('/ML-A100/team/mm/models/vit-base')
#     fixed_config = clip_model.vision_model.config

#     def __init__(self, opts, *args, **kwargs):
#         # 使用固定的 config
#         config = OpenViTModel.fixed_config

#         # 初始化 BaseImageEncoder
#         BaseImageEncoder.__init__(self, opts, *args, **kwargs)
        
#         # 初始化 CLIPVisionTransformer
#         CLIPVisionTransformer.__init__(self, config)
    
#     def get_activation_checkpoint_submodule_class(self) -> Callable:
#         """Returns the activation checkpoint module class.

#         For ViT, the activation checkpoint module class is TransformerEncoder or FlashTransformerEncoder.
#         """
#         return FlashTransformerEncoder
