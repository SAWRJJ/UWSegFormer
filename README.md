# UWSegFormer
More information will be available when the paper is published
## Requirements
 * Python 3.6+
 * Pytorch 1.3+
 * mmcv-full>=1.3.17, <1.6.0 (we use mmcv 1.5.3 and mmsegmentation 0.11.0 in code)
## Test
  `python tools/test.py configs/_our_/water_r50_fpn_1x.py model_checkpoint_path --eval segm`
