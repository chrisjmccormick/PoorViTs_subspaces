# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
from typing import Optional

import gin
import torch
import torch.nn as nn
from .mlp import Mlp, MlpDecomp

from utils.utils_ssl import trunc_normal_
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

fused_attn = hasattr(F, 'scaled_dot_product_attention')
print(f'Using fused_attn: {fused_attn}')

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,num_groups=32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features        
        self.k = nn.Parameter(torch.randn(in_features//num_groups, num_groups, hidden_features,requires_grad=True))
        self.v = nn.Parameter(torch.randn(in_features//num_groups, num_groups, hidden_features,requires_grad=True))
        self.act = act_layer()
        self.num_groups = num_groups
        
        self.proj = nn.Linear(in_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        q = (x).reshape(B,N, self.num_groups, C // self.num_groups)
        k = (self.k)
        
        score = self.act(torch.einsum('b n g d, d g l -> b n g l',q, k)) # B N g l
        # normalize the scores
        attn = F.softmax(score, dim=-1)
        # attn = attn / attn.sum(dim=-1, keepdim=True)  best so far
        x = torch.einsum('b n g l, c g s -> b n (g c)', attn, self.v)
        x = self.proj(x)
        x = self.drop(x)
        return x

@gin.configurable
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 q_proj_dim=None, kv_proj_dim=None,
                 output_subspace=False, o_proj_dim=None, o_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        all_head_dim = head_dim * self.num_heads

        q_proj_dim = q_proj_dim or dim
        kv_proj_dim = kv_proj_dim or dim

        # Shared query and key-value latent projections
        self.q_proj = nn.Linear(dim, q_proj_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, kv_proj_dim, bias=qkv_bias)

        # Per-head projections from latent spaces
        self.q = nn.Linear(q_proj_dim, all_head_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_proj_dim, 2 * all_head_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.output_subspace = output_subspace
        if self.output_subspace:
            if o_proj_dim is None:
                o_proj_dim = dim
            self.proj = nn.Linear(all_head_dim, o_proj_dim, bias=o_bias)
            self.o_proj = nn.Linear(o_proj_dim, dim, bias=False)
        else:
            self.proj = nn.Linear(all_head_dim, dim, bias=o_bias)
            self.o_proj = None

        self.proj_drop = nn.Dropout(proj_drop)

        

    def forward(self, x):
        B, N, C = x.shape
        
        q_latent = self.q_proj(x)
        kv_latent = self.kv_proj(x)

        q = self.q(q_latent)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(kv_latent)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(2)

        if fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0., 
                scale=self.scale
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v    

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        if self.o_proj is not None:
            x = self.o_proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/cait.py#L24
    # with slight modifications to do CA

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 num_cls_token=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_cls_token = num_cls_token
        self.scale = qk_scale if qk_scale is not None else head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.projO = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, o = x.split([self.num_cls_token, N-self.num_cls_token], dim=1)
        q = self.q(q).reshape(B, self.num_cls_token, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if fused_attn:
            x_cls = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_cls = attn @ v

        x_cls = x_cls.transpose(1, 2).reshape(B, self.num_cls_token, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        x_o = self.projO(o)
        
        return torch.cat([x_cls, x_o], dim=1)
        

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, init_values=None, attn_layer=Attention,
                 mlp_block=Mlp, q_proj_dim=None, kv_proj_dim=None,
                 output_subspace=False, o_proj_dim=None, o_bias=True):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            q_proj_dim=q_proj_dim, kv_proj_dim=kv_proj_dim,
            output_subspace=output_subspace, o_proj_dim=o_proj_dim, o_bias=o_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()

    def forward(self, x, return_attention=False):
        
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

@gin.configurable('ViT')
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0,
                 embed_dim=192, depth=12, num_heads=12, mlp_ratio=2.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 num_cls_token=1, depth_token_only=0,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, sin_pos=False, mlp_block=Mlp,
                 q_proj_dim=None, kv_proj_dim=None,
                 output_subspace=False, o_proj_dim=None, o_bias=True,
                 **kwargs):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.num_cls_token = num_cls_token

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls_token, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                mlp_block=mlp_block,
                q_proj_dim=q_proj_dim, kv_proj_dim=kv_proj_dim,
                output_subspace=output_subspace, o_proj_dim=o_proj_dim, o_bias=o_bias)

            for i in range(depth)])
        self.depth_token_only = depth_token_only
        self.blocks_token_only = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,

                attn_layer=partial(ClassAttn, num_cls_token=num_cls_token), mlp_block=mlp_block,
                q_proj_dim=q_proj_dim, kv_proj_dim=kv_proj_dim,
                output_subspace=output_subspace, o_proj_dim=o_proj_dim, o_bias=o_bias)
                attn_layer=partial(ClassAttn, num_cls_token=num_cls_token), mlp_block=mlp_block)

            for i in range(depth_token_only)])
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim*self.num_cls_token, num_classes) if num_classes > 0 else nn.Identity()
        if sin_pos:
            self.build_2d_sincos_position_embedding()
        else:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)



    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = 8,8
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False



    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        B, L, _  = x.shape

        #assert mask is not None
        # create mask tokens for patches
        # if L < 196:
        #     mask_token = self.mask_token.expand(B, 36, -1)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x0 = torch.cat((cls_tokens, x), dim=1)
        # add positional encoding to each token
        x = x0 + self.interpolate_pos_encoding(x0, w, h)       
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for i,blk in enumerate(self.blocks):
            x = blk(x)
        
        for i,blk in enumerate(self.blocks_token_only):
            x = blk(x)
        
        x = self.norm(x)
        
        z = x[:,:self.num_cls_token].flatten(1)
        
        return self.head(z)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

if __name__ == '__main__':
    model = VisionTransformer()
    x = torch.randn(10, 3, 64, 64)
    y = model(x)
    print(y.shape)
    print(y)
