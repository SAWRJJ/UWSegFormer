# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

from .Enblock import *

#########################################################

#########################################################
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#########################################################
class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.BatchNorm2d,
                 act=None,
                 bias_attr=False):
        super(ConvBNAct, self).__init__()
        # print("segformer_head.py--50", in_channels,"segformer_head.py--50", out_channels)
        # b0 in_channels 32 out_channels 128
        # b1 in_channels 64 out_channels 128
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.act = act() if act is not None else nn.Identity()
        self.bn = norm(out_channels) \
            if norm is not None else nn.Identity()

    def forward(self, x):
        # print("segformer_head.py--63", x.shape)
        # b1 torch.Size([1, 32, 80, 60])
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



# Flops: 2.84 GFLOPs
# Params: 3.51 M
class InjectionMultiSumallmultiallsum(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 384],  # [32, 64, 160, 256],
                 activations=None,
                 out_channels=256):  # 128
        super(InjectionMultiSumallmultiallsum, self).__init__()
        self.embedding_list = nn.ModuleList()
        self.act_embedding_list = nn.ModuleList()
        self.act_list = nn.ModuleList()
        for i in range(len(in_channels)):
            # print("segformer_head--102",in_channels[i])
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i], out_channels, kernel_size=1))
            self.act_list.append(activations())

    def forward(self, inputs):  # x_x8, x_x16, x_x32, x_x64
        # print("segformer_head--146", inputs[3].shape)
        # b1 inputs.shape[0] torch.Size([1, 32, 160, 120])
        low_feat1 = F.interpolate(inputs[0], scale_factor=0.5, mode="bilinear")
        # 有问题

        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")
        low_feat2_act = self.act_list[1](
            self.act_embedding_list[1](low_feat2))  # x16
        low_feat2 = self.embedding_list[1](low_feat2)

        high_feat_act = F.interpolate(
            self.act_list[2](self.act_embedding_list[2](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat2_act = F.interpolate(
            self.act_list[3](self.act_embedding_list[3](inputs[3])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat2 = F.interpolate(
            self.embedding_list[3](inputs[3]),
            size=low_feat2.shape[2:],
            mode="bilinear")
        # low_feat1_act x4
        # low_feat2_act x8
        res = low_feat1_act * low_feat2_act * high_feat_act * (
                low_feat1 + low_feat2) + high_feat * high_feat2_act + high_feat2

        return res

@HEADS.register_module()
class MAA_segformer1(BaseDecodeHead):
    def __init__(self,
                 feature_strides,
                 # in_index,
                 # num_classes,
                 # in_channels,
                 use_dw=False,
                 dropout_ratio=0.1,
                 # out_feat_chs=None,
                 act_layer=nn.Sigmoid,
                 # channels=256,
                 align_corners=False,
                 **kwargs
                 ):
        super(MAA_segformer1, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.align_corners = align_corners
        self.last_channels = self.channels

        # print("segformer_head--144",self.in_channels,"segformer_head--144",self.channels)
        # b0+PP self.in_channels [32, 64, 160, 256] self.channels 128
        # b0 self.in_channels [32, 64, 160, 256] self.channels 128

        #
        # b1+PP self.in_channels [64, 128, 320, 512] self.channels 128
        self.inj_module = InjectionMultiSumallmultiallsum(
            in_channels=self.in_channels,
            activations=act_layer,
            out_channels=self.channels)
        # print("segformer--163")
        self.linear_fuse = ConvBNAct(
            in_channels=self.last_channels,
            out_channels=self.last_channels,
            kernel_size=1,
            stride=1,
            groups=self.last_channels if use_dw else 1,
            act=nn.ReLU)

        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(
            self.last_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        # segformer_head--327 torch.Size([1, 32, 160, 120])
        # segformer_head--328 torch.Size([1, 64, 80, 60])
        # segformer_head--327 torch.Size([1, 160, 40, 30])
        # segformer_head--328 torch.Size([1, 256, 20, 15])
        # print("segformer_head--327",x[0].shape)
        # print("segformer_head--328", x[1].shape)
        # print("segformer_head--327",x[2].shape)
        # print("segformer_head--328", x[3].shape)

        x = self.inj_module(x)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        # print("segformer_head--345",x.shape)
        # print(1/0)
        return x


#########################################################
@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        # print("segformer_head--144",self.in_channels,"segformer_head--144",self.channels)
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32

        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
