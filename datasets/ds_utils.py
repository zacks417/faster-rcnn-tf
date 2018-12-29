from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# 返回唯一框的所言
def unique_boxes(boxes,scale=1.0):
    v = np.array([1,1e3,1e6,1e9])
    hashes = np.round(boxes*scale).dot(v)
    # 去除重复元素，并将元素从小到大返回一个新的列表，返回新列表元素在旧列表中的位置
    _,index = np.unique(hashes,return_index=True)
    return np.sort(index)

def xywh_to_xyxy(boxes):
    # [x y w h ] 转为 [ x1 y1 x2 y2]
    return np.hstack((boxes[:,0:2],boxes[:,0:2]+boxes[:,2:4]-1))

def xyxy_toxywh(boxes):
    return np.hstack((boxes[:,0:2],boxes[:,2:4]-boxes[:,0:2]+1))

def validate_boxes(boxes,width=0,height=0):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    assert (x1>=0).all()
    assert (y1>=0).all()
    assert (x2>=x1).all()
    assert (y2>=y1).all()
    assert (x2<width).all()
    assert (y2<height).all()

def filter_small_boxes(boxes,min_size):
    w = boxes[:,2] - boxes[:,0]
    h = boxes[:,3] - boxes[:,1]
    keep = np.where((w>=min_size)&(h>min_size))[0]
    return keep
