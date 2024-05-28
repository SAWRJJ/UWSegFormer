# -*- coding: utf-8 -*-
# time: 2023/2/23 15:59
# file: SUIM.py
# author: SAW

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SUIMDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        "BW", "HD", "WR", "RO", "RI", "FV")

    PALETTE = [[0, 0, 0], [0, 0, 255],
               [0, 255, 255], [255, 0, 0], [255, 0, 255],
               [255, 255, 0]]
    # CLASSES = (
    #     "HD", "WR", "RO", "RI", "FV")
    #
    # PALETTE = [[0, 0, 255],
    #            [0, 255, 255], [255, 0, 0], [255, 0, 255],
    #            [255, 255, 0]]

    def __init__(self, **kwargs):
        super(SUIMDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            # reduce_zero_label=True,
            **kwargs)


