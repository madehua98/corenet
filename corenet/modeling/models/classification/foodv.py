""" ViTamin

Paper: Designing Scalable Vison Models in the Vision-Language Era

@misc{chen2023designing,
      title={Designing Scalable Vison Models in the Vision-Language Era},
      author={Jieneng Chen and Qihang Yu and Xiaohui Shen and Alan Yuille and Liang-Cheih Chen},
      year={2023},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Based on Apache 2.0 licensed code at https://github.com/ViTamin/ViTamin

Modifications and timm support by Jieneng Chen 2023

Adapted from timm codebase, thanks!
"""

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
from corenet.modeling.models.classification.config.foodv import get_configuration
from corenet.modeling.modules import FlashTransformerEncoder, TransformerEncoder
from corenet.utils import logger

import torch
import torch.nn as nn
from typing import List, Tuple

class DenseConnector:
    def __init__(self, 
                 layer_indices_sti: Tuple[int, int] = (7, 15), 
                 layer_indices_sci: Tuple[int, int] = (7, 15), 
                 layer_indices_dci: Tuple[int, int, int] = (0, 12, 24), 
                 mlp_depth: int = 2):
        self.layer_indices_sti = layer_indices_sti
        self.layer_indices_sci = layer_indices_sci
        self.layer_indices_dci = layer_indices_dci
        self.mlp_depth = mlp_depth

    def dense_connector_sti(self, image_features: torch.Tensor, image_forward_outs: List[torch.Tensor]) -> torch.Tensor:
        image_features_1 = image_forward_outs[self.layer_indices_sti[0]].to(image_features.dtype).contiguous()
        return image_features_1

    def dense_connector_sci(self, image_features: torch.Tensor, image_forward_outs: List[torch.Tensor]) -> torch.Tensor:
        image_features_1 = image_forward_outs[self.layer_indices_sci[0]].to(image_features.dtype).contiguous()
        return image_features_1

    def dense_connector_dci(self, image_features: torch.Tensor, image_forward_outs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features_1 = []
        image_features_2 = []
        for i in range(self.layer_indices_dci[0], self.layer_indices_dci[1]):
            image_features_1.append(image_forward_outs[i].to(image_features.dtype).contiguous())
        image_features_1 = torch.stack(image_features_1, dim=0)
        image_features_1 = torch.sum(image_features_1, dim=0) / (self.layer_indices_dci[1] - self.layer_indices_dci[0])
        for i in range(self.layer_indices_dci[1], self.layer_indices_dci[2]):
            image_features_2.append(image_forward_outs[i].to(image_features.dtype).contiguous())
        image_features_2 = torch.stack(image_features_2, dim=0)
        image_features_2 = torch.sum(image_features_2, dim=0) / (self.layer_indices_dci[2] - self.layer_indices_dci[1])
        return image_features_1, image_features_2

    def dense_connector(self, image_features: torch.Tensor, image_features_dc: torch.Tensor, mm_dense_connector_type: str = 'sti') -> torch.Tensor:
        if mm_dense_connector_type == 'sti':
            image_features = torch.cat((image_features, image_features_dc), dim=-2)
        elif mm_dense_connector_type == 'sci':
            image_features = torch.cat((image_features, image_features_dc), dim=-2)
        elif mm_dense_connector_type == 'dci':
            image_features = torch.cat((image_features, image_features_dc), dim=-2)
        else:
            raise NotImplementedError()
        return image_features


def mlp_layer(mlp_depth, input_size, hidden_size):
    modules = [nn.Linear(input_size, hidden_size)]
    for _ in range(1, mlp_depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)


@dataclass
class VitConvCfg:
    expand_ratio: float = 4.0
    expand_output: bool = True  # calculate expansion channels from output (vs input chs)
    kernel_size: int = 3
    group_size: int = 1  # 1 == depthwise
    pre_norm_act: bool = False  # activation after pre-norm
    stride_mode: str = 'dw'  # stride done via one of 'pool', '1x1', 'dw'
    pool_type: str = 'avg2'
    downsample_pool_type: str = 'avg2'
    act_layer: str = 'gelu' # stem & stage 1234
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None
    down_shortcut: Optional[bool] = True
    mlp: str = 'mlp'

    def __post_init__(self):
        use_mbconv = True
        if not self.norm_layer:
            self.norm_layer = 'batchnorm2d' if use_mbconv else 'layernorm2d'
        if not self.norm_layer_cl and not use_mbconv:
            self.norm_layer_cl = 'layernorm'
        if self.norm_eps is None:
            self.norm_eps = 1e-5 if use_mbconv else 1e-6
        self.downsample_pool_type = self.downsample_pool_type or self.pool_type

@dataclass
class VitCfg:
    embed_dim: Tuple[Union[int, Tuple[int, ...]], ...] = (96, 192, 384, 768)
    depths: Tuple[Union[int, Tuple[int, ...]], ...] = (2, 3, 5, 2)
    stem_width: int = 64
    conv_cfg: VitConvCfg = field(default_factory=VitConvCfg)
    weight_init: str = 'vit_eff'
    head_type: str = ""
    stem_type: str = "stem"

def _init_conv(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            bias: bool = True,
    ):
        super().__init__()
        self.grad_checkpointing=False
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs
        self.conv1 = create_conv2d(in_chs, out_chs, 3, stride=2, bias=bias)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = create_conv2d(out_chs, out_chs, 3, stride=1, bias=bias)
        named_apply(_init_conv, self)

    def forward(self, x):
        if self.grad_checkpointing:
            x = checkpoint(self.conv1, x)
            x = self.norm1(x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.conv2(x)

        return x

class Downsample2d(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            bias: bool = True,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias) # 1x1 conv
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.expand(x)
        return x


class StridedConv(nn.Module):
    """ downsample 2d as well
    """
    def __init__(
            self, 
            kernel_size=3, 
            stride=2, 
            padding=1,
            in_chans=3, 
            embed_dim=768, 
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        norm_layer = partial(get_norm_layer('layernorm2d'), eps=1e-6)
        self.norm = norm_layer(in_chans) 

    def forward(self, x):
        x = self.norm(x) 
        x = self.proj(x)
        return x


class MbConvLNBlock(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            drop_path: float = 0.,
            kernel_size: int = 3,
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            act_layer: str = 'gelu',
            expand_ratio: float = 4.0,
    ):
        super(MbConvLNBlock, self).__init__()
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        if stride == 2: 
            self.shortcut = Downsample2d(in_chs, out_chs, bias=True)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)
        self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = _create_act(act_layer, inplace=True)
        self.act2 = _create_act(act_layer, inplace=True)

        self.conv2_kxk = create_conv2d(mid_chs, mid_chs, kernel_size, stride=stride, dilation=1, groups=mid_chs, bias=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def init_weights(self, scheme=''):
        named_apply(partial(_init_conv, scheme=scheme), self)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x) # nn.Identity() 

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x) 

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x


class MbConvStages(nn.Module):
    """ stage 1 and stage 2 of ViTamin: MBConv-LN blocks
    """
    def __init__(
            self,
            cfg: VitCfg,
            img_size: Union[int, Tuple[int, int]] = 224, # place holder
            in_chans: int = 3,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg.stem_width,
        )
        stages = []
        self.num_stages = len(cfg.embed_dim)
        for s, dim in enumerate(cfg.embed_dim[:2]):
            blocks = []
            stage_in_chs = cfg.embed_dim[s-1] if s>0 else cfg.stem_width
            for d in range(cfg.depths[s]):
                blocks += [MbConvLNBlock(
                        in_chs = stage_in_chs if d==0 else dim,
                        out_chs = dim,
                        stride = 2 if d == 0 else 1,
                    )]
            blocks = nn.Sequential(*blocks)
            stages += [blocks]

        self.stages = nn.ModuleList(stages)
        self.pool = StridedConv(
                        stride=2,
                        in_chans=cfg.embed_dim[1],
                        embed_dim=cfg.embed_dim[2]
                    )

    def forward(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for stage in self.stages:
                x = checkpoint_seq(stage, x)
            x = checkpoint(self.pool, x)
        else:
            for stage in self.stages:
                x = stage(x)
            x = self.pool(x)
        
        return x

class GeGluMlp(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features,
            act_layer = None,
            drop = 0.0,
    ):
        super().__init__()
        norm_layer = partial(get_norm_layer('layernorm'), eps=1e-6)
        self.norm = norm_layer(in_features)
        self.act = nn.GELU()
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class HybridEmbed(nn.Module):
    """ 
    Extract feature map from stage 1-2, flatten, project to embedding dim.
    """
    def __init__(
            self,
            backbone,
            img_size=224,
            patch_size=1,
            feature_size=None,
            in_chans=3,
            embed_dim=1024,
            bias=True,
            dynamic_img_pad=False,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        # if feature_size is None:
        #     feature_size = img_size[0] // 16
        feature_size = 16
        feature_size = to_2tuple(feature_size)
        if hasattr(self.backbone, 'feature_info'):
            feature_dim = self.backbone.feature_info.channels()[-1]
        elif hasattr(self.backbone, 'num_features'):
            feature_dim = self.backbone.num_features
        else:
            feature_dim = embed_dim
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Identity()

    def forward(self, x):
        x = self.backbone(x) #经过backbone为变为 [256,384,14,14]
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def _trunc_normal_(tensor, mean, std, a, b):
    # rewrite timm trunc normal
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated standard normal
    # tensor.erfinv_() # NOTE: deleted as "erfinv_cuda" not implemented for 'BFloat16'

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

@MODEL_REGISTRY.register(name="foodv", type="classification")
class Foodv(BaseImageEncoder):
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            opts,
            *args, **kwargs
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__(opts, *args, **kwargs)
        ViTamin_config = get_configuration(opts)
        n_class = getattr(
            self.opts, "model.classification.n_classes"
        )
        self.num_classes = n_class
        self.global_pool = ViTamin_config["global_pool"]  # 'token'
        self.class_token = ViTamin_config["class_token"]  # false
        self.num_prefix_tokens = 1 if self.class_token else 0
        self.reg_tokens = ViTamin_config["reg_tokens"]  # 0
        self.num_prefix_tokens += self.reg_tokens
        self.fc_norm = ViTamin_config["fc_norm"]  # None
        self.norm_layer = ViTamin_config["norm_layer"]  # Optional[LayerType]
        self.act_layer = ViTamin_config["act_layer"]  # Optional[LayerType]
        assert self.global_pool in ('', 'avg', 'token', 'map')  # token
        assert self.class_token or self.global_pool != 'token'  # True, token
        self.use_fc_norm = self.global_pool == 'avg' if self.fc_norm is None else self.fc_norm   #None
        self.norm_layer = get_norm_layer(self.norm_layer) or partial(nn.LayerNorm, eps=1e-6)   # None
        self.act_layer = get_act_layer(self.act_layer) or nn.GELU  # None

        self.use_flash_attn = ViTamin_config["use_flash_attn"] # False
        self.grad_checkpointing = ViTamin_config["grad_checkpointing"] # False
        self.img_size = ViTamin_config["img_size"]  # 256
        self.patch_size = ViTamin_config["patch_size"]  # 16
        self.in_chans = ViTamin_config["in_chans"]  # 3
        self.embed_dim = ViTamin_config["embed_dim"]  # 6C
        self.block1_embed_dim = ViTamin_config["block1_embed_dim"]  # 8C
        self.depth = ViTamin_config["depth"]  # 12
        self.num_heads = ViTamin_config["num_heads"]  # 384/32
        self.block1_num_heads = ViTamin_config["block1_num_heads"] # 
        self.mlp_ratio = ViTamin_config["mlp_ratio"]  # 4.0
        self.qkv_bias = ViTamin_config["qkv_bias"]  # True
        self.qk_norm = ViTamin_config["qk_norm"]  # False
        self.init_values = ViTamin_config["init_values"]  # None
        self.no_embed_class = ViTamin_config["no_embed_class"]  # False
        self.pre_norm = ViTamin_config["pre_norm"]  # False
        self.dynamic_img_size = ViTamin_config["dynamic_img_size"]  # False
        self.dynamic_img_pad = ViTamin_config["dynamic_img_pad"]  # False
        self.drop_rate = ViTamin_config["drop_rate"]  # 0.0
        self.pos_drop_rate = ViTamin_config["pos_drop_rate"]  # 0.0
        self.patch_drop_rate = ViTamin_config["patch_drop_rate"]  # 0.0
        self.proj_drop_rate = ViTamin_config["proj_drop_rate"]  # 0.0
        self.attn_drop_rate = ViTamin_config["attn_drop_rate"]  # 0.0
        self.drop_path_rate = ViTamin_config["drop_path_rate"]  # 0.0
        self.weight_init = ViTamin_config["weight_init"]  # ''
        self.fix_init = ViTamin_config["fix_init"]  # False
        #self.embed_layer = ViTamin_config["embed_layer"]  # PatchEmbed          #使用PatchEmbed和ViTamin提出的MbConv进行对比
        self.block_fn = ViTamin_config["block_fn"]  # Type[nn.Module]
        self.mlp_layer = GeGluMlp  # Type[nn.Module]
        self.is_pos_embed = ViTamin_config["is_pos_embed"]  # True
        self.MbConv_embed_dim = ViTamin_config["MbConv_embed_dim"] # [64, 128, 384]
        self.MbConv_depths = ViTamin_config["MbConv_depths"] # [2,4,1]
        self.MbConv_stem_width = ViTamin_config["MbConv_stem_width"] # 64
        self.mm_dense_connector_type = ViTamin_config["mm_dense_connector_type"]
        
        self.use_kl = False
        self.model_conf_dict = {
            "conv1": {"in": 3, "out": self.block1_embed_dim},
            "layer1": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
            "layer2": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
            "layer3": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
            "layer4": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
            "layer5": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
            "exp_before_cls": {"in": self.block1_embed_dim, "out": self.block1_embed_dim},
        }
        embed_args = {}
        if self.dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))

        stage_1_2 = MbConvStages(cfg=VitCfg(
            embed_dim=(self.MbConv_embed_dim[0], self.MbConv_embed_dim[1], self.MbConv_embed_dim[2]),
            depths=(self.MbConv_depths[0], self.MbConv_depths[1], self.MbConv_depths[2]),
            stem_width=self.MbConv_stem_width,
            conv_cfg = VitConvCfg(
                norm_layer='layernorm2d',
                norm_eps=1e-6,
            ),
            head_type='1d',
            ), 
        )
        self.patch_embed = HybridEmbed(
            stage_1_2,
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            bias=not self.pre_norm,
            dynamic_img_pad=self.dynamic_img_pad,
            **embed_args,)
        # self.patch_embed = embed_layer(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        #     bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        # )
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if self.class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, self.reg_tokens, self.embed_dim)) if self.reg_tokens else None
        
        if self.is_pos_embed:
            embed_len = num_patches if self.no_embed_class else num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)
        else:
            self.pos_embed = None
        
        self.pos_drop = nn.Dropout(p=self.patch_drop_rate)
        if self.patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                self.patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = self.norm_layer(self.embed_dim) if self.pre_norm else nn.Identity()

        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            self.block_fn(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_norm=self.qk_norm,
                init_values=self.init_values,
                proj_drop=self.proj_drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
            )
            for i in range(self.depth)])
        
        self.pool = StridedConv(
                        stride=2,
                        in_chans=self.embed_dim,
                        embed_dim=self.block1_embed_dim
                    )
        
        self.blocks1 = nn.Sequential(*[
            self.block_fn(
                dim=self.block1_embed_dim,
                num_heads=self.block1_num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_norm=self.qk_norm,
                init_values=self.init_values,
                proj_drop=self.proj_drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
            )
            for i in range(self.depth)])
        
        self.norm = self.norm_layer(self.block1_embed_dim) if not self.use_fc_norm else nn.Identity()

        # connecter
        # self.block_to_block1 = LinearLayer(self.embed_dim, self.block1_embed_dim)

        self.dense_connnector = DenseConnector(layer_indices_sti=(self.depth-1,2 * self.depth - 1), 
                                              layer_indices_sci=(self.depth-1, 2 * self.depth - 1),
                                              layer_indices_dci=(0, self.depth, 2 * self.depth),
                                              )
        
        self.mlp = mlp_layer(mlp_depth = 2, input_size= self.block1_embed_dim, hidden_size=self.block1_embed_dim)

        # Classifier Head
        if self.global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.block1_embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                norm_layer=self.norm_layer,
            )
        else:
            self.attn_pool = None

        self.fc_norm = self.norm_layer(self.block1_embed_dim) if self.use_fc_norm else nn.Identity()
        self.classifier_drop = nn.Dropout(self.drop_rate)
        self.classifier = LinearLayer(self.block1_embed_dim, self.num_classes)

        # if self.weight_init != 'skip':
        #     self.init_weights(self.weight_init)
        # if self.fix_init:
        #     self.fix_init_weight()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == Foodv:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--model.classification.foodv.mode",
                type=str,
                default="small",
                choices=["small", "base", "large", "huge"],
                help="vitamin mode. Default is base.",
            )
            group.add_argument(
                "--model.classification.foodv.connector_type",
                type=str,
                default="dci",
                choices=["sci", "dci"],
                help="vitamin connector_type",
            )
            group.add_argument(
                "--model.classification.foodv.dropout",
                type=float,
                default=0.0,
                help="Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.foodv.stochastic-dropout",
                type=float,
                default=0.0,
                help="Stochastic Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.foodv.norm-layer",
                type=str,
                default="layer_norm",
                help="Normalization layer to be used in Transformer layer. Defaults to LayerNorm.",
            )

            group.add_argument(
                "--model.classification.foodv.sinusoidal-pos-emb",
                action="store_true",
                default=False,
                help="Use sinusoidal instead of learnable positional embedding. Defaults to False.",
            )
            group.add_argument(
                "--model.classification.foodv.no-cls-token",
                action="store_true",
                default=False,
                help="Do not use classification token. Defaults to False.",
            )

            group.add_argument(
                "--model.classification.foodv.use-simple-fpn",
                action="store_true",
                default=False,
                help="Add simple FPN for down-stream tasks (e.g., detection). Defaults to False.",
            )
            group.add_argument(
                "--model.classification.foodv.use-flash-attention",
                action="store_true",
                default=False,
                help="Use Transformer layers with flash attention for efficiently computing scaled dot-product attention. Defauls to False.",
            )

        return parser
        
    
    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpoint module class.

        For ViT, the activation checkpoint module class is TransformerEncoder or FlashTransformerEncoder.
        """
        return FlashTransformerEncoder if self.use_flash_attn else TransformerEncoder
    
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if self.is_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.is_pos_embed:
            return {'pos_embed', 'cls_token', 'dist_token'}
        else:
            return {'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        self.patch_embed.backbone.stem.grad_checkpointing = enable # disable https://blog.csdn.net/lhx526080338/article/details/127894671?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-127894671-blog-125562110.235^v38^pc_relevant_anti_t3_base&spm=1001.2101.3001.4242.2&utm_relevant_index=4
        self.patch_embed.backbone.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _pos_embed(self, x):
        if self.no_embed_class:
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def pool_stage3_stage4(self, x):
        x = x.transpose(1, 2)
        n, c, h_w = x.shape
        h = w = int(h_w ** 0.5)
        x = x.view(n, c, h, w)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.is_pos_embed:
            x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
            x = self.pool_stage3_stage4(x)
            x = self.blocks1(x)
        x = self.norm(x)
        return x

    # def forward_features_dense_connector(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.patch_embed(x)
    #     if self.is_pos_embed:
    #         x = self._pos_embed(x)
    #     x = self.patch_drop(x)
    #     x = self.norm_pre(x)
    #     all_hidden_states = []
    #     if self.grad_checkpointing and not torch.jit.is_scripting():
    #         x = checkpoint_seq(self.blocks, x)
    #     else:
    #         average_stage3 = torch.zeros()
    #         for block in self.blocks:
    #             x = block(x)
    #             average_x
    #             all_hidden_states.append(x)
    #         x = self.pool_stage3_stage4(x)

    #         for block1 in self.blocks1:
    #             x = block1(x)
    #             all_hidden_states.append(x)
    #     x = self.norm(x)
    #     return x, all_hidden_states

    def forward_classifier(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.classifier_drop(x)
        return x if pre_logits else self.classifier(x)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     if self.neural_augmentor is not None:
    #         out_dict = {"augmented_tensor": None}
    #         if self.training and self.neural_augmentor is not None:
    #             x = self.neural_augmentor(x)
    #             out_dict.update({"augmented_tensor": x})
    #         x, all_hidden_states = self.forward_features_dense_connector(x)
    #         if self.mm_dense_connector_type == 'sci':
    #             image_features_dc = self.dense_connnector.dense_connector_sci(x, all_hidden_states)
    #             image_features_dc = self.pool_stage3_stage4(image_features_dc)
    #             x = torch.cat((x, image_features_dc), dim=-2)

    #         elif self.mm_dense_connector_type == 'dci':
    #             image_features_dc1, image_features_dc2 = self.dense_connnector.dense_connector_dci(x, all_hidden_states)
    #             image_features_dc1 = self.pool_stage3_stage4(image_features_dc1)
    #             x = torch.cat((image_features_dc1, image_features_dc2), dim=-2)
    #         x = self.mlp(x)
    #         logits = self.forward_classifier(x)
    #         out_dict.update({"logits": logits})
    #         return out_dict
    #     else:
    #         logits, _ = self.forward_classifier(x)
    #         return logits

    def forward_features_dense_connector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.is_pos_embed:
            x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        sum_x = torch.zeros_like(x)  # 初始化为和x相同形状和device的全0张量
        count = 0
        for block in self.blocks:
            x = block(x)
            sum_x += x  # 累计求和
            count += 1  # 统计次数
            
        average_x = sum_x / count  # 计算均值
        x = self.pool_stage3_stage4(x)

        sum_x1 = torch.zeros_like(x)  # 第二阶段也同样处理
        count1 = 0
        for block1 in self.blocks1:
            x = block1(x)
            sum_x1 += x
            count1 += 1
            
        average_x1 = sum_x1 / count1  # 计算第二个阶段的均值
        return x, average_x, average_x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.neural_augmentor is not None:
            out_dict = {"augmented_tensor": None}
            if self.training and self.neural_augmentor is not None:
                x = self.neural_augmentor(x)
                out_dict.update({"augmented_tensor": x})
            _, average_stage3, average_stage4 = self.forward_features_dense_connector(x)
            average_stage3 = self.pool_stage3_stage4(average_stage3)
            x = torch.cat((average_stage3, average_stage4), dim=-2)
            x = self.mlp(x)
            logits = self.forward_classifier(x)
            out_dict.update({"logits": logits})
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

        x, average_stage3, average_stage4 = self.forward_features_dense_connector(x)
        #x = average_stage4
        x = x.transpose(1, 2)
        n, c, h_w = x.shape
        h = w = int(h_w ** 0.5)
        x = x.view(n, c, h, w)
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


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    return build_model_with_cfg(
        ViTamin, # ViTamin
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )


def _create_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_vision_transformer(variant, pretrained=pretrained, embed_layer=embed_layer, **kwargs)


@register_model
def vitamin_small(pretrained=False, **kwargs) -> VisionTransformer:
    stage_1_2 = MbConvStages(cfg=VitCfg(
            embed_dim=(64, 128, 384),
            depths=(2, 4, 1),
            stem_width=64,
            conv_cfg = VitConvCfg(
                norm_layer='layernorm2d',
                norm_eps=1e-6,
            ),
            head_type='1d',
        ),
    )
    stage3_args = dict(embed_dim=384, depth=14, num_heads=6, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid('vitamin_small', backbone=stage_1_2, pretrained=pretrained, **dict(stage3_args, **kwargs))
    return model


@register_model
def vitamin_base(pretrained=False, **kwargs) -> VisionTransformer:
    stage_1_2 = MbConvStages(cfg=VitCfg(
            embed_dim=(128, 256, 768),
            depths=(2, 4, 1),
            stem_width=128,
            conv_cfg = VitConvCfg(
                norm_layer='layernorm2d',
                norm_eps=1e-6,
            ),
            head_type='1d',
        ),
    )
    stage3_args = dict(embed_dim=768, depth=14, num_heads=12, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid('vitamin_base', backbone=stage_1_2, pretrained=pretrained, **dict(stage3_args, **kwargs))
    return model


@register_model
def vitamin_large(pretrained=False, **kwargs) -> VisionTransformer:
    stage_1_2 = MbConvStages(cfg=VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    stage3_args = dict(embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_large', backbone=stage_1_2, pretrained=pretrained, **dict(stage3_args, **kwargs))
    return model


@register_model
def vitamin_large_256(pretrained=False, **kwargs) -> VisionTransformer:
    backbone = MbConvStages(cfg=VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    model_args = dict(img_size=256, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_large_256', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_336(pretrained=False, **kwargs) -> VisionTransformer:
    backbone = MbConvStages(cfg=VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    model_args = dict(img_size=336, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_large_336', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_large_384(pretrained=False, **kwargs) -> VisionTransformer:
    backbone = MbConvStages(cfg=VitCfg(
        embed_dim=(160, 320, 1024),
        depths=(2, 4, 1),
        stem_width=160,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    model_args = dict(img_size=384, embed_dim=1024, depth=31, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, is_pos_embed=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_large_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_xlarge_256(pretrained=False, **kwargs) -> VisionTransformer:
    backbone = MbConvStages(cfg=VitCfg(
        embed_dim=(192, 384, 1152),
        depths=(2, 4, 1),
        stem_width=192,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    model_args = dict(img_size=256, embed_dim=1152, depth=32, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, is_pos_embed=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_xlarge_256', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitamin_xlarge_384(pretrained=False, **kwargs) -> VisionTransformer:
    backbone = MbConvStages(cfg=VitCfg(
        embed_dim=(192, 384, 1152),
        depths=(2, 4, 1),
        stem_width=192,
        conv_cfg = VitConvCfg(
            norm_layer='layernorm2d',
            norm_eps=1e-6,
        ),
        head_type='1d',
    ), 
    )
    model_args = dict(img_size=384, embed_dim=1152, depth=32, num_heads=16, mlp_layer=GeGluMlp, mlp_ratio=2., class_token=False, is_pos_embed=False, global_pool='avg')
    model = _create_vision_transformer_hybrid(
        'vitamin_xlarge_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


def count_params(model: nn.Module):
    return sum([m.numel() for m in model.parameters()])


def count_stage_params(model: nn.Module, prefix='none'):
    collections = []
    for name, m in model.named_parameters():
        print(name)
        if name.startswith(prefix):
            collections.append(m.numel())
    return sum(collections)


if __name__ == "__main__":
    model = timm.create_model('vitamin_large', num_classes=10).cuda()
    # x = torch.rand([2,3,224,224]).cuda()
    check_keys(model)

