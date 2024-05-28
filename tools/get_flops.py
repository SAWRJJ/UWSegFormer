# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import time

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info
import sys

sys.path.insert(0, 'D:\PythonV6\SegFormer-master')
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    # dict_keys(['meta', 'state_dict', 'optimizer'])
    # "work_dirs/segformer.b0.640x480.suim.160k(fwq)/iter_120000.pth"
    # "work_dirs/segformer_e+PPsegHead1.b0.640x480.suim.160k(81.68)/iter_136000.pth"
    # work_dirs/segformer_e+PPsegHead1.b0.640x480.dut.40k/iter_16000.pth # FLOPS：3.21G  FPS: 58.22
    # work_dirs/segformer_e+PPsegHead1.b0.640x480.dut.40k256/iter_32000.pth # Flops: 3.65 GFLOPs FPS: 52.7537 fps
    # work_dirs/segformer_e+PPsegHead1.b0.640x480.dut.40k512/iter_20000.pth # Flops: 4.99 GFLOPs FPS: 53.8267 fps
    # work_dirs/segformer.b0.640x480.suim.160k(fwq)/iter_120000.pth Flops: 7.94 GFLOPs Params: 3.72 M
    # average fps: 95.97246883523404
    # slowest fps: 49.74859447277903
    # work_dirs/segformer_e.b0.640x480.suim.160k(fwq)/iter_68000.pth  Flops: 8.31 GFLOPs Params: 22.02 M
    # average fps: 58.185595544810205
    # slowest fps: 30.993157466932683
    # work_dirs/segformer+PPsegHead1.b0.640x480.suim.160k（fwq）/iter_156000.pth Flops: 2.84 GFLOPs Params: 3.47 M
    # average fps: 141.21474937377113
    # slowest fps: 71.0272979746664
    # work_dirs/segformer_e+PPsegHead1.b0.640x480.suim.160k(81.68)/iter_136000.pth Flops: 3.21 GFLOPs Params: 21.78 M
    # average fps: 62.32332047184576
    # slowest fps: 38.583568675430286
    # work_dirs/segformer.b0.640x480.dut.40k（fwq）/iter_36000.pth
    # DUT 62.25

    # e1+PP1 Flops: 7.05 GFLOPs
    # Params: 17.44 M

    # DUT UNet FPS: 18.4600 fps
    # Flops: 238.9 GFLOPs
    # Params: 29.07 M

    # SUIM UNet FPS: 20.5592 fps
    # Flops: 237.94 GFLOPs
    # Params: 29.06 M
    # content = torch.load("work_dirs/fcn_unet_s5-d16_640x480_160k_suim/iter_160000.pth", map_location=torch.device('cpu'))
    # print(content['state_dict'].keys())
    # model.load_state_dict(content['state_dict'], False)
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)

    t_all = []
    print(args.shape)
    total_time = 0
    # pic_file = 'data/SUIM/images/validation/'
    # paths = []
    # for filename in os.listdir(pic_file):
    #     path = pic_file + filename
    #     paths.append(path)
    #     image = cv2.imread(path)
    #     # 转换图像颜色空间为RGB
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # 转换图像为张量
    #     x = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
    #     x = x.unsqueeze(0).cuda()
    #     t1 = time.time()
    #     with torch.no_grad():
    #         model(x)
    #     t2 = time.time()
    #     total_time += t2 - t1
    #     t_all.append(t2 - t1)
    # average_time = total_time / len(paths)

    image_path = "data/SUIM/images/validation/d_r_47_.jpg"
    # image_path = "data/DUT/images/validation/4283.jpg"
    image = cv2.imread(image_path)

    # 转换图像颜色空间为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换图像为张量
    x = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
    x = x.unsqueeze(0).cuda()
    # x = torch.bernoulli(torch.ones(1, 3, args.shape[0], args.shape[1])).cuda()
    for i in range(100):
        t1 = time.time()
        with torch.no_grad():
            model(x)
        t2 = time.time()
        total_time += t2 - t1
        t_all.append(t2 - t1)

    average_time = total_time / 100
    print("Average inference time: {:.4f} seconds".format(average_time))
    print("FPS: {:.4f} fps".format(1 / average_time))
    print('average time:', np.mean(t_all) / 1)
    print('average fps:', 1 / np.mean(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:', 1 / max(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:', 1 / min(t_all))

    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
