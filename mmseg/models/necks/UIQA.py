from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from PIL import Image
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

############################################################

import copy

import math
import torch
import torch.nn as nn

from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

from ..builder import NECKS


class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, patchsize, in_channels, H, W):
        super().__init__()
        # img_size = _pair(img_size)
        self.patch_size = _pair(patchsize)
        # print("CMSFFT--34--img_size",img_size)
        # print("CMSFFT--35--patch_size",patch_size)
        self.in_channels = in_channels
        # print("CMSFFT--38",n_patches) # 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=self.patch_size,
                                       stride=self.patch_size,
                                       # padding=3
                                       )
        # 可训练参数 维度为(1,n_p,in_c)
        n_patches = (H // self.patch_size[0]) * (W // self.patch_size[1])
        # device = torch.device('cuda:0')
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        # print("CMSFFT--68", x.shape)  # CMSFFT--52 torch.Size([1, 32, 128, 128])

        x = self.patch_embeddings(x)  # (B, hidden，n_patches^(1/2), n_patches^(1/2))
        # print("CMSFFT--71", x.shape)  # CMSFFT--54 torch.Size([1, 32, 4, 4])
        # 压缩维度 表示从dim这个维度开始，进行最大程度的压缩
        x = x.flatten(2)
        # transpose()：转置（两个维度）
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print("CMSFFT--49--x", x.shape)
        # CMSFFT--49--x torch.Size([1, 16, 32])
        # CMSFFT--49--x torch.Size([1, 16, 64])
        # CMSFFT--49--x torch.Size([1, 16, 160])
        # CMSFFT--49--x torch.Size([1, 16, 256])

        # print("CMSFFT--82--self.position_embeddings", position_embeddings.shape)
        # CMSFFT--49--x torch.Size([1, 15, 32])
        # CMSFFT--50--self.position_embeddings torch.Size([1, 16, 32])
        embeddings = x + self.position_embeddings
        # embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


# 特征重组
class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        # self.H = img_size[0]
        # self.W = img_size[1]

    def forward(self, x, H, W, full_h, full_w):
        if x is None:
            return None

        # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # print("CMSFFT--93", x.size())  # CMSFFT--88 torch.Size([1, 15, 32])
        # CMSFFT --88 torch.Size([1, 15, 32])
        # CMSFFT --88 torch.Size([1, 15, 64])
        # CMSFFT --88 torch.Size([1, 15, 160])
        # CMSFFT --88 torch.Size([1, 15, 256])
        B, n_patch, hidden = x.size()
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        h, w = H, W
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(size=(full_h, full_w))(x)
        # print("CMSFFT--104", x.size())
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


# Channel-wise Attention
class Attention_org(nn.Module):
    def __init__(self, vis, channel_num, KV_size=480, num_heads=4):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = KV_size
        self.channel_num = channel_num
        self.num_attention_heads = num_heads

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(num_heads):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            key = nn.Linear(self.KV_size, self.KV_size, bias=False)
            value = nn.Linear(self.KV_size, self.KV_size, bias=False)
            # 把所有的值都重新复制一遍，deepcopy为深复制，完全脱离原来的值，即将被复制对象完全再复制一遍作为独立的新个体单独存在
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        # Instance Norm
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        # Softmax
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        return O1, O2, O3, O4, weights


class MlpE(nn.Module):
    def __init__(self, in_channel, mlp_channel):
        super(MlpE, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.0)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# channel wise muti-head attention
class Block_ViT(nn.Module):
    def __init__(self, vis, channel_num, expand_ratio=4, KV_size=480):
        super(Block_ViT, self).__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.attn_norm = LayerNorm(KV_size, eps=1e-6)
        self.channel_attn = Attention_org(vis, channel_num, KV_size=KV_size)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        self.ffn1 = MlpE(channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = MlpE(channel_num[1], channel_num[1] * expand_ratio)
        self.ffn3 = MlpE(channel_num[2], channel_num[2] * expand_ratio)
        self.ffn4 = MlpE(channel_num[3], channel_num[3] * expand_ratio)

    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4
        for i in range(4):
            var_name = "emb" + str(i + 1)  # emb1,emb2,emb3,emb4
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)
        # print("CMSFFT--293", embcat[0].shape)
        # print("CMSFFT--294", embcat[1].shape)
        # print("CMSFFT--295", embcat[2].shape)
        # print("CMSFFT--296", embcat[3].shape)
        # CMSFFT--293 torch.Size([1, 16, 32])
        # CMSFFT--293 torch.Size([1, 16, 64])
        # CMSFFT--293 torch.Size([1, 16, 160])
        # CMSFFT--293 torch.Size([1, 16, 256])
        emb_all = torch.cat(embcat, dim=2)
        # print("CMSFFT--302", emb_all.shape)
        # CMSFFT--302 torch.Size([1, 16, 512])
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1, cx2, cx3, cx4, emb_all)
        # 残差
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None

        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None
        # 残差
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, vis, channel_num, num_layers=4, KV_size=512):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        for _ in range(num_layers):
            layer = Block_ViT(vis, channel_num, KV_size=KV_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4, weights = layer_block(emb1, emb2, emb3, emb4)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, emb3, emb4, attn_weights

@NECKS.register_module()
class UIQA(nn.Module):
    def __init__(self, vis=False, img_size=[256, 256], channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4],
                 KV_size=512,pretrained=None,init_cfg=None,frozen = False):
        super().__init__()
        self.full_h = img_size[0]
        self.full_w = img_size[1]
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]

        self.embeddings_1 = Channel_Embeddings(self.patchSize_1,
                                               # img_size=[img_size, img_size],
                                               # img_size=img_size,
                                               in_channels=channel_num[0],
                                               H=self.full_h,
                                               W=self.full_w)
        self.embeddings_2 = Channel_Embeddings(self.patchSize_2,
                                               # img_size=[img_size // 2, img_size // 2],
                                               # img_size=img_size // 2,
                                               in_channels=channel_num[1],
                                               H=self.full_h // 2,
                                               W=self.full_w // 2
                                               )
        self.embeddings_3 = Channel_Embeddings(self.patchSize_3,
                                               # img_size=[img_size // 4, img_size // 4],
                                               # img_size=img_size // 4,
                                               in_channels=channel_num[2],
                                               H=self.full_h // 4,
                                               W=self.full_w // 4
                                               )
        self.embeddings_4 = Channel_Embeddings(self.patchSize_4,
                                               # img_size=[img_size // 8, img_size // 8],
                                               # img_size=img_size // 8,
                                               in_channels=channel_num[3],
                                               H=self.full_h // 8,
                                               W=self.full_w // 8
                                               )

        self.encoder = Encoder(vis, channel_num, KV_size=KV_size)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1),
                                         # img_size=img_size
                                         )
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2),
                                         # img_size=img_size // 2
                                         )
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,
                                         scale_factor=(self.patchSize_3, self.patchSize_3),
                                         # img_size=img_size // 4
                                         )
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,
                                         scale_factor=(self.patchSize_4, self.patchSize_4),
                                         # img_size=img_size // 8
                                         )
        self.pretrained = pretrained
        # assert not (init_cfg and pretrained), \
        #     'init_cfg and pretrained cannot be setting at the same time'
        # if isinstance(pretrained, str):
        #     warnings.warn('DeprecationWarning: pretrained is deprecated, '
        #                   'please use "init_cfg" instead')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        # elif pretrained is None:
        #     if init_cfg is None:
        #         self.init_cfg = [
        #             dict(type='Kaiming', layer='Conv2d'),
        #             dict(
        #                 type='Constant',
        #                 val=1,
        #                 layer=['_BatchNorm', 'GroupNorm'])
        #         ]
        # else:
        #     raise TypeError('pretrained must be a str or None')

        self.init_weights()
        self.apply(self._init_weights)

        self.frozen = frozen
        if self.frozen:
            for name, param in self.named_parameters():
                # print(name)
                param.requires_grad = False
        # print(1/0)

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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False, logger=logger)

    def forward(self, x):
        en1, en2, en3, en4 = x[0],x[1],x[2],x[3]
        # H, W = self.full_h // 4, self.full_w // 4,
        H, W = self.full_h, self.full_w,
        # print("CMSFFT--446", en1.shape)  # torch.Size([1, 32, 128, 128])
        emb1 = self.embeddings_1(en1)
        # print("CMSFFT--381",emb1.shape) # torch.Size([1, 16, 32])
        # emb2 = self.embeddings_2(en2, H // 2, W // 2)
        # emb3 = self.embeddings_3(en3, H // 4, W // 4)
        # emb4 = self.embeddings_4(en4, H // 8, W // 8)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        # emb1 = en1
        # emb2 = en2
        # emb3 = en3
        # emb4 = en4

        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1, emb2, emb3,
                                                                            emb4)  # (B, n_patch, hidden)
        # print("CMSFFT--420",encoded1.shape) # CMSFFT--420 torch.Size([1, 15, 32])
        x1 = self.reconstruct_1(encoded1, H // self.patchSize_1,
                                W // self.patchSize_1, self.full_h, self.full_w) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2, H // (self.patchSize_2 * 2),
                                W // (self.patchSize_2 * 2), self.full_h // 2,
                                self.full_w // 2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3, H // (self.patchSize_3 * 4),
                                W // (self.patchSize_3 * 4), self.full_h // 4,
                                self.full_w // 4) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4, H // (self.patchSize_4 * 8),
                                W // (self.patchSize_4 * 8), self.full_h // 8,
                                self.full_w // 8) if en4 is not None else None
        # print("CMSFFT--468", x1.shape)
        # print("CMSFFT--469", en1.shape)
        # CMSFFT--439 torch.Size([2, 32, 640, 480])
        # CMSFFT--440 torch.Size([2, 32, 160, 120])
        x1 = x1 + en1 if en1 is not None else None
        # print("CMSFFT--473", x2.shape)
        # print("CMSFFT--474", en2.shape)
        # CMSFFT--473 torch.Size([2, 64, 160, 120])
        # CMSFFT--474 torch.Size([2, 64, 80, 60])
        x2 = x2 + en2 if en2 is not None else None
        x3 = x3 + en3 if en3 is not None else None
        x4 = x4 + en4 if en4 is not None else None

        return x1, x2, x3, x4, attn_weights

# if __name__ == '__main__':
#     mtc = ChannelTransformer(
#         img_size=enhance_img_size // 4,
#         channel_num=embed_dims,
#         patchSize=patchsize, KV_size=KV_szie)