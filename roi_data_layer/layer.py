from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
    def __init__(self,roidb,num_classes,random=False):
        self._roidb = roidb
        self._num_classes = num_classes
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000))%4294967295
            np.random.seed(millis)
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            # 宽度大于高度位置
            horz = (widths >= heights)
            # 高度大于宽度位置
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            # 随机打乱数组 再重组(这里打乱 并不会破坏原来的顺序，只是重新生成
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds,(-1,2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm,:],(-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        if self._random:
            np.random.set_state(st0)
        self._cur = 0

    def _get_next_minibatch_inds(self):
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur:self._cur+cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db,self._num_classes)

    def forward(self):
        blobs = self._get_next_minibatch()
        return blobs
