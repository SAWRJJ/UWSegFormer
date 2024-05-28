# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


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

        # print("CMSFFT--82--self.position_embeddings", self.position_embeddings.shape)
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
    def __init__(self, vis, channel_num, KV_size=480, num_heads=2):
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
        # output
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
        # print("mix_e--239",context_layer4.shape) # mix_e--239 torch.Size([8, 15, 256, 4])
        # print(1/0)
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


class UIQA(nn.Module):
    def __init__(self, vis=False, img_size=[256, 256], channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4],
                 KV_size=512):
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

    def forward(self, en1, en2, en3, en4):
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# dim 相当于 channel
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # print("mix_transformer.py--526", x.shape)
        # mix_transformer.py--430 torch.Size([1, 19200, 32])
        # mix_transformer.py--430 torch.Size([1, 19200, 32])
        # mix_transformer.py--430 torch.Size([1, 4800, 64])
        # mix_transformer.py--430 torch.Size([1, 4800, 64])
        # mix_transformer.py--430 torch.Size([1, 1200, 160])
        # mix_transformer.py--430 torch.Size([1, 1200, 160])
        # mix_transformer.py--430 torch.Size([1, 300, 256])
        # mix_transformer.py--430 torch.Size([1, 300, 256])
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.ca = CA_Block(dim=dim)
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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # print("mix_transformer.py--486", x.size())
        # Mix-FFM 经过Mix-FFN维度不变
        """
        mix_transformer.py--487 torch.Size([8, 19200, 32])
        mix_transformer.py--487 torch.Size([8, 19200, 32])
        mix_transformer.py--487 torch.Size([8, 4800, 64])
        mix_transformer.py--487 torch.Size([8, 4800, 64])
        mix_transformer.py--487 torch.Size([8, 1200, 160])
        mix_transformer.py--487 torch.Size([8, 1200, 160])
        mix_transformer.py--487 torch.Size([8, 300, 256])
        mix_transformer.py--487 torch.Size([8, 300, 256])
        """
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # print("mix_transformer.py--491", x.size())
        # x = x + self.drop_path(self.ca(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        # print("mix_transformer.py--543",x.shape)
        _, _, H, W = x.shape
        # flatten()是对多维数据的降维函数
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # print("mix_transformer.py--546",x.shape)
        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], patchSize=[32, 16, 8, 4], KV_size=512,
                 enhance_img_size=[224, 224]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.img_size = np.array(img_size)
        self.patchsize = patchSize
        self.KV_szie = KV_size
        self.embed_dims = embed_dims
        self.enhance_img_size = np.array(enhance_img_size)
        # self.enhance = enhanceNet(in_chans=in_chans, img_size=enhance_img_size)
        self.mtc = UIQA(
            img_size=self.enhance_img_size // 4,
            channel_num=embed_dims,
            patchSize=self.patchsize, KV_size=self.KV_szie)
        # # 图像增强模块
        # self.enhence_layer = FeatureEnhancementGatingModule(dim=embed_dims[0], img_size=img_size, patch_size=patch_size,
        #                                                     stride=1,
        #                                                     in_chans=in_chans, embed_dim=embed_dims[0], )
        # patch_embed

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def print_img_size(self):
        print(self.img_size)
        print(self.embed_dims)

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

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):

        B = x.shape[0]
        h, w = x.shape[2], x.shape[3]

        outs = []
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        outs0, outs1, outs2, outs3, _ = self.mtc(outs[0], outs[1], outs[2], outs[3])
        # 残差
        outs[0] = outs[0] + outs0 if outs0 is not None else None
        outs[1] = outs[1] + outs1 if outs1 is not None else None
        outs[2] = outs[2] + outs2 if outs2 is not None else None
        outs[3] = outs[3] + outs3 if outs3 is not None else None

        return outs

    def forward(self, x):
        # self.print_img_size()
        x = self.forward_features(x)

        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


@BACKBONES.register_module()
class mit_b0_e(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0_e, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1,
            patchSize=[32, 16, 8, 4],
            KV_size=512,
            enhance_img_size=[480, 640]
            # 480, 640
            # 960, 1280
        )

#
# @BACKBONES.register_module()
# class mit_b1_e(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b1_e, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1,
#             patchSize=[32, 16, 8, 4],
#             KV_size=1024,
#             enhance_img_size=[480, 640]
#             # 480, 640
#         )
#
#
# @BACKBONES.register_module()
# class mit_b2_e(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b2_e, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#
#
# @BACKBONES.register_module()
# class mit_b3_e(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b3_e, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)
#
#
# @BACKBONES.register_module()
# class mit_b4_e(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b4_e, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1,
#             patchSize=[32, 16, 8, 4],
#             KV_size=1024,
#             enhance_img_size=[480, 640]
#         )
#
#
# @BACKBONES.register_module()
# class mit_b5_e(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b5_e, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1,
#             patchSize=[32, 16, 8, 4],
#             KV_size=1024,
#             enhance_img_size=[480, 640]
#         )
