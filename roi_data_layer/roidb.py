from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps
from PIL import Image

# imdb 默认保存的roidb包含boxes,gt_overlaps,gt_classes和flipped
# 再进行扩充：保存图片地址，图片的尺寸，最大的overlap以及对应类别
def prepare_roidb(imdb):
    roidb = imdb.roidb
    if not imdb.name.startswith('coco'):
        sizes = [Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)]
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        if not imdb.name.startswith('coco'):
            roidb[i]['width'] = sizes[i][0]
            roidb[i]['height'] = sizes[i][1]
        # 转为数组 一会求max argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps

        zero_inds = np.where(max_overlaps==0)[0]
        assert all(max_classes[zero_inds] == 0)
        nonzero_inds = np.where(max_overlaps>0)[0]
        assert all(max_classes[nonzero_inds]!=0)

