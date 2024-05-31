# UWSegFormer
More information will be available when the paper is published
## Requirements
 * Python 3.6+
 * Pytorch 1.3+
 * mmcv-full>=1.3.17, <1.6.0 (we use mmcv 1.5.3 and mmsegmentation 0.11.0 in code)
## Dataset
Please create a data folder in your working directory and put the SUIM dataset or DUT dataset in it for testing.
```
data
  ├── SUIM
  |   ├── annotations
  │   │   ├── trainging
  |   |   │   ├── ......
  │   │   ├── validation
  |   |   │   ├── ......
  │   ├── images
  │   │   ├── trainging
  |   |   │   ├── ......
  │   │   ├── validation
  |   |   │   ├── ......
```

## Test
  Download trained weights.  Evaluate UWSegFormer on SUIM:  
  `python tools/test.py local_configs/uwsegformer/uw.b0.640x480.suim.160k.py model_checkpoint_path --eval mIoU`  
  or evaluate UWSegFormer on DUT:  
  `python tools/test.py local_configs/uwsegformer/uw.b0.640x480.dut.40k.py model_checkpoint_path --eval mIoU`  
  Note that all checkpoints and datasets are available at [Google Drive](https://drive.google.com/drive/u/0/folders/1-qtLVDZzWg62khOEaPZXmM3I_8noA4WW)

