from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
    import cProfile as pickle
except ImportError:
    import pickle
import os
import math

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg,get_output_dir
from model.bbox_transform import clip_boxes,bbox_transform_inv
from nms.nms import nms

def _get_image_blob(im):
    # 将图片转为网络输入
    im_orig = im.astype(np.float32,copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # 防止最大的维度超过最大尺寸
        if np.round(im_scale*im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float













