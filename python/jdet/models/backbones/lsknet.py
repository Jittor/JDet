import jittor as jt
import jittor.nn as nn
from jittor import init
import math
from functools import partial
from jdet.utils.registry import BACKBONES
from jittor.nn import DropPath
from jittor.init import trunc_normal_
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------
import errno
import hashlib
import os
import requests
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from tqdm import tqdm

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

def to_2tuple(x):
    if isinstance(x, (int, float)):
        return (x, x)
    elif isinstance(x, (tuple, list)) and len(x) == 2:
        return tuple(x)
    else:
        raise ValueError("Input must be an int, float, or a tuple/list of length 2.")



def _is_legacy_zip_format(filename):
    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False

def _legacy_zip_load(filename, model_dir, map_location):
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return jt.load(extracted_file)


model_urls = {
    "lsknet_t": "https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_t_backbone-2ef8a593.pth",
    "lsknet_s": "https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_s_backbone-e9d2e551.pth"
}

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10, 'input_size': (3, 1024, 1024), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=1)

    def execute(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = jt.concat([attn1, attn2], dim=1)
        avg_attn = jt.mean(attn, dim=1, keepdims=True)
        max_attn = jt.max(attn, dim=1, keepdims=True)
        agg = jt.concat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    
class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def execute(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = layer_scale_init_value * jt.ones((dim))
        self.layer_scale_2 = layer_scale_init_value * jt.ones((dim))

        self.apply(self._init_weights)

    # def start_grad(x):
    #     return x._update(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=1024, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))

            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def execute(self, x):
        x = self.proj(x)
        # _, _, H, W = x.shape
        x = self.norm(x)        
        return x


class LSKNet(nn.Module):
    def __init__(self, img_size=1024, in_chans=3, num_classes=10, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False, out_indices = (0, 1, 2)):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.out_indices = out_indices
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            init.gauss_(m.weight, 0.0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                # m.bias.data.zero_()
                init.constant_(m.bias)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # 
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x = patch_embed(x)
            _, _, H, W = x.shape
            for blk in block:
                x = blk(x)
            # x = x.flatten(2).transpose(1, 2)
            x = jt.transpose(x.flatten(2),0,2,1)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)#.contiguous()
            if i in self.out_indices:
                outs.append(x)
        return outs

    def execute(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def execute(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
def LSKNet_t(pretrained=False, **kwargs):
    model = LSKNet(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=nn.LayerNorm, depths=[3, 3, 5, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        model.load(pretrained)
    return model

@BACKBONES.register_module()
def LSKNet_s(pretrained=False, **kwargs):
    model = LSKNet(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=nn.LayerNorm, depths=[2, 2, 4, 2],
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        model.load(pretrained)
    return model

