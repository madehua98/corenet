#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict

from corenet.utils import logger
from typing import Callable, Optional, Union, Tuple, List, Sequence
from timm.models.vision_transformer import Mlp, Block, PatchEmbed, VisionTransformer
from dataclasses import dataclass, replace, field

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

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
    act_layer1: str = 'gelu' # stage 1234
    act_layer2: str = 'gelu' # stage 1234
    norm_layer: str = ''
    norm_layer_cl: str = ''
    norm_eps: Optional[float] = None
    down_shortcut: Optional[bool] = True
    mlp: str = 'mlp'

    def __post_init__(self):
        # mbconv vs convnext blocks have different defaults, set in post_init to avoid explicit config args
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
    # embed_dim: Tuple[int, ...] = (96, 192, 384, 768)
    embed_dim: Tuple[Union[int, Tuple[int, ...]], ...] = (96, 192, 384, 768)
    depths: Tuple[Union[int, Tuple[int, ...]], ...] = (2, 3, 5, 2)
    stem_width: int = 64
    conv_cfg: VitConvCfg = field(default_factory=VitConvCfg)
    weight_init: str = 'vit_eff'
    head_type: str = ""
    stem_type: str = "stem"
    ln2d_permute: bool = True


def get_configuration(opts: argparse.Namespace) -> Dict:

    mode = getattr(opts, "model.classification.foodv.mode", "tiny")
    if not mode:
        logger.error("Please specify mode")

    mode = mode.lower()
    print(mode)
    dropout = getattr(opts, "model.classification.foodv.dropout", 0.0)
    norm_layer = getattr(opts, "model.classification.foodv.norm_layer", "layer_norm")

    ViTamin_config = {
        "qkv_bias": True,
        "qk_norm": False,
        "init_values": None,
        "class_token": False,
        "no_embed_class": False,
        "grad_checkpointing": False,
        "reg_tokens": 0,
        "pre_norm": False,
        "fc_norm": None,
        "dynamic_img_size": False,
        "dynamic_img_pad": False,
        "use_flash_attn": False,
        "drop_rate": 0.0,
        "pos_drop_rate": 0.0,
        "patch_drop_rate": 0.0,
        "proj_drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "weight_init": '',
        "fix_init": False,
        "embed_layer": PatchEmbed,
        "norm_layer": None,
        "act_layer": None,
        "block_fn": Block,
        "mlp_layer": Mlp,
        "is_pos_embed": True,
        "mm_dense_connector_type": 'dci'
    }
    if mode == "small":
        ViTamin_config.update({
            "img_size": 224,
            "patch_size": 14,
            "in_chans": 3,
            "global_pool": 'avg',
            "embed_dim": 384,
            "block1_embed_dim": 512,
            "depth": 8,
            "num_heads": 6,
            "block1_num_heads": 8,
            "mlp_ratio": 4.0,
            "MbConv_embed_dim": [64, 128, 384],
            "MbConv_depths": [2, 4, 1],
            "MbConv_stem_width": 64,
        })
    elif mode == "base":
        ViTamin_config.update({
            "img_size": 224,
            "patch_size": 14,
            "in_chans": 3,
            "global_pool": 'avg',
            "embed_dim": 768,
            "block1_embed_dim": 1024,
            "depth": 8,
            "num_heads": 12,
            "block1_num_heads": 16,
            "mlp_ratio": 4.0,
            "MbConv_embed_dim": [128, 256, 768],
            "MbConv_depths": [2, 4, 1],
            "MbConv_stem_width": 128,
        })
    elif mode == "large":
        foodv = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "large-256":
        foodv = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "large-336":
        foodv = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "large-384":
        foodv = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "large-256":
        foodv = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }



    else:
        logger.error("Got unsupported ViT configuration: {}".format(mode))
    return ViTamin_config
