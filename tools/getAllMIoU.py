# -*- coding: utf-8 -*-
# time: 2023/5/16 7:16
# file: getAllMIoU.py
# author: SAW
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
import sys

# work_dirs\segformer+PPsegHead.b0.640x480.suim.160k\segformer+PPsegHead.b0.640x480.suim.160k.py
# work_dirs\segformer+PPsegHead.b0.640x480.suim.160k\iter_156000.pth
# --eval
# mIoU
sys.path.insert(0, 'D:\PythonV6\SegFormer-master')
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('all_file', help='include all files')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(args1):
    args = args1

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = False

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # print(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    # print(dataset.__getitem__(0))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    ck = []
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True  # False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            return dataset.evaluate(outputs, args.eval, **kwargs)['mIoU']
            # dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == '__main__':
    # main()
    args1 = parse_args()
    pic_file = args1.all_file
    # print(pic_file)
    # work_dirs\\segformer+PPsegHead.b0.640x480.suim.160k\\iter_156000.pth
    mious = {}
    for filename in os.listdir(pic_file):
        if filename.endswith(".pth") and filename != 'latest.pth':
            img_path = os.path.join(pic_file, filename)
            args = parse_args()
            # args.checkpoint = 'hahaha'
            # print(img_path[1:])
            # print(img_path)
            args.checkpoint = img_path
            mious[filename] = main(args)
    print(sorted(mious.items(), key=lambda x: x[1], reverse=True))
    # best SUIM （PP）('iter_136000.pth', 0.7775)
    # best SUIM (origin)
    # best SUIM （PP1）('iter_104000.pth', 0.8113) ('iter_60000.pth', 0.8056)
    # best SUIM （eV2+PP1）('iter_124000.pth', 0.818) ('iter_124000.pth', 0.8148000000000001)
    # best SUIM (eV3+PP1) ('iter_120000.pth', 0.8131)
    # best DUT （PP）('iter_52000.pth', 0.7079)
    # best DUT （eV2+PP1）('iter_12000.pth', 0.701)
    # best DUT （eV3+PP1）('iter_24000.pth', 0.6969)

    # 放到论文中的结果
    # SUIM (segformer)(640_3080Ti): /('iter_120000.pth', 0.8057)/,('iter_148000.pth', 0.8028),('iter_156000.pth', 0.7996), ('iter_108000.pth', 0.799)
    # SUIM (segformer+PP1)(640_3080Ti): /('iter_156000.pth', 0.8104)/, ('iter_160000.pth', 0.8099)
    # SUIM (segformer_e)(640_3080Ti): /('iter_68000.pth', 0.8076000000000001)/
    # SUIM (segformer_e+PP1)(640_3070Ti): /('iter_136000.pth', 0.8184)/ (160000.pth, 82.12)
    # SUIM (_e1+PP1): ('iter_108000.pth', 0.8170000000000001)



    # SUIM (N2): /('iter_136000.pth', 0.8058)/
    # SUIM (N3): ('iter_156000.pth', 0.8101)
    # SUIM (N5): ('iter_148000.pth', 0.81)
    # SUIM (N6): ('iter_152000.pth', 0.8062)

    # L
    # SUIM (L2): ('iter_144000.pth', 0.8151)
    # SUIM (L3):81.72
    # L4       :82.12
    # SUIM (L5):
    # SUIM (L6): 152,81.32
    # SUIM (L7):

    # csa head
    # SUIM (n1): Flops: 3.1 GFLOPs # Params: 14.33 M
    # SUIM (n2): /('iter_128000.pth', 0.8125)/ Flops: 3.14 GFLOPs # Params: 16.81 M
    # SUIM (n3): ('iter_128000.pth', 0.8148000000000001), ('iter_112000.pth', 0.8145) Flops: 3.17 GFLOPs # Params: 19.3 M
    # SUIM (n5): ('iter_60000.pth', 0.8178) Flops: 3.25 GFLOPs # Params: 24.26 M
    # SUIM (n6): [('iter_156000.pth', 0.7914)] Flops: 3.29 GFLOPs # Params: 26.74 M

    # SUIM (DeepLabV3+_MobilenetV2): ('iter_160000.pth', 0.7103)
    # SUIM (PSPNet+MobilenetV2): ('iter_160000.pth', 0.6976)
    # SUIM (FCN+MV2): ('iter_160000.pth', 0.6640999999999999),
    # SUIM (UNet): ('iter_160000.pth', 0.5611999999999999)

    # SUIM (LVT): ('iter_116000.pth', 0.8072)
    # +-------+-------+-------+
    # | Class | IoU | Acc |
    # +-------+-------+-------+
    # | BW | 88.24 | 91.33 |
    # | HD | 84.49 | 93.52 |
    # | WR | 76.84 | 87.58 |
    # | RO | 81.56 | 85.62 |
    # | RI | 70.12 | 90.0 |
    # | FV | 83.12 | 87.94 |
    # +-------+-------+-------+
    # Summary:
    #
    # +--------+-------+-------+-------+
    # | Scope | mIoU | mAcc | aAcc |
    # +--------+-------+-------+-------+
    # | global | 80.72 | 89.33 | 90.54 |
    # +--------+-------+-------+-------+

    # SUIM (LVT+PP1): ('iter_108000.pth', 0.8099)
    # SUIM (LVT_e):/('iter_68000.pth', 0.8142)/
    # 68000
    # +-------+-------+-------+
    # | Class | IoU | Acc |
    # +-------+-------+-------+
    # | BW | 88.8 | 92.65 |
    # | HD | 82.97 | 92.79 |
    # | WR | 77.23 | 85.3 |
    # | RO | 85.14 | 88.02 |
    # | RI | 72.53 | 89.51 |
    # | FV | 81.85 | 88.27 |
    # +-------+-------+-------+

    # DUT (segformer)(640_3080Ti): ('iter_32000.pth', 0.7261),('iter_16000.pth', 0.7121),/('iter_36000.pth', 0.7081999999999999)/, ('iter_24000.pth', 0.6679)
    # DUT (segformer+PP1)(640_3080Ti): ('iter_36000.pth', 0.7245999999999999),('iter_40000.pth', 0.7148), /('iter_32000.pth', 0.7109000000000001)/,
    # DUT (segformer_e)(640_3080Ti): ('iter_32000.pth', 0.7145999999999999), /('iter_20000.pth', 0.7126)/, ('iter_16000.pth', 0.7120000000000001)
    # DUT (segformer_e+PP1)(640_3070Ti): ('iter_32000.pth', 0.6995999999999999)
    # DUT (segformer_e+PP1)(640_3080Ti): /('iter_16000.pth', 0.7123)/ 20000.pth, 0.7141
    # DUT (e+PP1+N2): ('iter_20000.pth', 0.7186), ('iter_16000.pth', 0.7141), ('iter_12000.pth', 0.7128), ('iter_24000.pth', 0.7037)
    # DUT (e+PP1+N3): ('iter_16000.pth', 0.7058)
    # DUT (N5):('iter_32000.pth', 0.7036), ('iter_40000.pth', 0.7015), ('iter_28000.pth', 0.7006999999999999)
    # DUT (N6):('iter_28000.pth', 0.7158), ('iter_40000.pth', 0.7086)

    # DUT (segformer_e+PP1_256)(640_3080Ti): /('iter_32000.pth', 0.7128)/, ('iter_24000.pth', 0.7082), ('iter_40000.pth', 0.7053)
    # DUT (segformer_e+PP1_512)(640_3080Ti): /('iter_20000.pth', 0.7149)/, ('iter_32000.pth', 0.7006), ('iter_40000.pth', 0.6904)
    # DUT b5: ('iter_16000.pth', 0.7379000000000001), ('iter_8000.pth', 0.7338), ('iter_36000.pth', 0.73)
    # DUT (_e1+PP1):('iter_32000.pth', 0.7208)

    # DUT (DLV3+MV2): ('iter_40000.pth', 0.6668999999999999)
    # DUT (FCN+MV2): ('iter_36000.pth', 0.6275999999999999)
    # DUT (PSP+MV2): ('iter_32000.pth', 0.6784)
    # DUT (UNet): ('iter_40000.pth', 0.5915)

    # UFO (b0): ('iter_84000.pth', 0.6178)
    # UFO (e+PP1):('iter_132000.pth', 0.6126)

    # for i in range(3):
    #     args = parse_args()
    #     # args.checkpoint = 'hahaha'
    #     print(args.checkpoint[66:])
