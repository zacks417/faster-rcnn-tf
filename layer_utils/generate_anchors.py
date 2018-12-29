from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def generate_anchors(base_size=16,ratios=[0.5,1,2],scales=2**np.arange(3,6)):
    base_anchor = np.array([1,1,base_size,base_size]) - 1
    # 生成不同比例的anchor
    ratio_anchors = _ratio_enum(base_anchor,ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i,:],scales) for i in range(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    # 返回一个anchor窗口的宽，高，xy中心
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5*(w-1)
    y_ctr = anchor[1] + 0.5*(h-1)
    return w,h,x_ctr,y_ctr

def _mkanchors(ws,hs,x_ctr,y_ctr):
    # 给定一组w h x y 输出一组anchors
    # 增加一维
    ws = ws[:,np.newaxis]
    hs = hs[:,np.newaxis]
    anchors = np.hstack((x_ctr-0.5*(ws-1),
                         y_ctr-0.5*(hs-1),
                         x_ctr+0.5*(ws-1),
                         y_ctr+0.5*(hs-1)))
    return anchors

def _ratio_enum(anchor,ratios):
    # 用anchor给每个纵横比枚举一组anchor点
    #  获取anchor的参数
    w,h,x_ctr,y_ctr = _whctrs(anchor)
    size = w*h
    size_ratios = size/ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws*ratios)
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

def _scale_enum(anchor,scales):
    # 每个anchor的每个比例枚举一组anchor
    w,h,x_ctr,y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws,hs,x_ctr,y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time()-t)
    print(a)
