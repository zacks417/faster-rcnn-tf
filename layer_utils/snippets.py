from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height,width,feat_stride,anchor_scales=(8,16,32),anchor_ratios=(0.5,1,2)):
    # 一个包装函数，用于生成给定不同比例的锚点 并返回anchors数量
    anchors = generate_anchors(ratios=np.array(anchor_ratios),scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0,width) * feat_stride
    shift_y = np.arange(0,height) * feat_stride
    # 接受两个一维数组生成两个二维矩阵 维度均为 y.shape x x.shape
    # shift_x 每一行均为shift_x,shift_y每一列均为shift_y
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    shifts = np.vstack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # 这里是 H W C
    anchors = anchors.reshape((1,A,4)) + shifts.reshape((1,K,4)).transpose((1,0,2))
    anchors = anchors.reshape((K*A,4)).astype(np.float32,copy=False)
    length = np.int32(anchors.shape[0])
    return anchors,length

def generate_anchors_pre_tf(height,width,feat_stride=16,anchor_scales=(8,16,32),anchor_ratios=(0.5,1,2)):
    shift_x = tf.range(width) * feat_stride
    shift_y = tf.range(height)*feat_stride
    shift_x,shift_y = tf.meshgrid(shift_x,shift_y)
    sx = tf.reshape(shift_x,shape=(-1,))
    sy = tf.reshape(shift_y,shape=(-1,))
    shifts = tf.transpose(tf.stack([sx,sy,sx,sy]))
    K = tf.multiply(width,height)
    shifts = tf.transpose(tf.reshape(shifts,shape=[1,K,4]),perm=(1,0,2))
    anchors = generate_anchors(ratios=np.array(anchor_ratios),scales=np.array(anchor_scales))
    A = anchors.shape[0]
    anchor_constant = tf.constant(anchors.reshape((1,A,4)),dtype=tf.int32)
    length = K * A
    anchors_tf = tf.reshape(tf.add(anchor_constant,shifts),shape=(length,4))

    return tf.cast(anchors_tf,dtype=tf.float32),length
