from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
    import cPickle as pickle
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
    # im BGR顺序
    im_orig = im.astype(np.float32,copy = True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # 600
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # 防止最大的维度超过最大尺寸 1000
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE)/float(im_size_max)
        im = cv2.resize(im_orig,None,None,fx=im_scale,fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    blob = im_list_to_blob(processed_ims)
    return blob,np.array(im_scale_factors)

def _get_blobs(im):
    # 将图片和rois转为网络输入
    blobs = {}
    blobs['data'],im_scale_factors = _get_image_blob(im)
    return blobs,im_scale_factors

def _clip_boxes(boxes,im_shape):
    # 将盒子裁剪到边界内,高在前
    boxes[:,0::4] = np.maximum(boxes[:,0::4],0)
    boxes[:,1::4] = np.maximum(boxes[:,1::4],0)
    boxes[:,2::4] = np.minimum(boxes[:,2::4],im_shape[1]-1)
    boxes[:,3::4] = np.minimum(boxes[:,3::4],im_shape[0]-1)
    return boxes

















