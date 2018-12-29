from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import numpy as np
from collections import namedtuple
from nets.networks import Network
from model.config import cfg

def separable_conv2d_same(inputs,kernel_size,stride,rate=1,scope=None):
    if stride == 1:
        # depth_multiplier: 卷积乘子，即每个输入通道经过卷积后的输出通道数。
        return slim.separable_conv2d(inputs,None,kernel_size,
                                     depth_multiplier=1,stride=1,rate=rate,
                                     padding='SAME',scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size-1)*(rate-1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total //2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0,0],[pad_beg,pad_end],[pad_beg,pad_end],[0,0]])
        return slim.separable_conv2d(inputs,None,kernel_size,
                                     depth_multiplier=1,stride=stride,rate=rate,
                                     padding='VALID',scope=scope)

Conv = namedtuple('Conv',['kernel','stride','depth'])
DepthSepConv = namedtuple('DepthSepConv',['kernel','stride','depth'])

_CONV_DEFS = [
    Conv(kernel=3,stride=2,depth=32),
    DepthSepConv(kernel=3,stride=1,depth=64),
    DepthSepConv(kernel=3,stride=2,depth=128),
    DepthSepConv(kernel=3,stride=1,depth=128),
    DepthSepConv(kernel=3,stride=2,depth=256),
    DepthSepConv(kernel=3,stride=1,depth=256),
    DepthSepConv(kernel=3,stride=2,depth=512),
    DepthSepConv(kernel=3,stride=1,depth=512),
    DepthSepConv(kernel=3,stride=1,depth=512),
    DepthSepConv(kernel=3,stride=1,depth=512),
    DepthSepConv(kernel=3,stride=1,depth=512),
    DepthSepConv(kernel=3,stride=1,depth=512),
    # 这里改为stride=1,是为了使得特征图总stride也为16
    DepthSepConv(kernel=3,stride=1,depth=1024),
    DepthSepConv(kernel=3,stride=1,depth=1024)
]
# 修改过的mobilenet_v1
def mobilenet_v1_base(inputs,
                      conv_defs,
                      starting_layer=0,
                      min_depth=8,
                      depth_multiplier=1.0,
                      output_stride=None,
                      reuse=None,
                      scope=None):
    # starting_layer 对于RPN为0,对于region classification 是12
    depth =lambda d:max(int(d*depth_multiplier),min_depth)
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with tf.variable_scope(scope,'MobilenetV1',[inputs],reuse=reuse):
        current_stride = 1
        rate = 1
        net = inputs
        for i, conv_def in enumerate(conv_defs):
            end_point_base = 'Conv2d_%d'%(i+starting_layer)
            if output_stride is not None and current_stride == output_stride:
                layer_stride = 1
                layer_rate = rate
                rate *= conv_def.stride
            else:
                layer_stride = conv_def.stride
                layer_rate = 1
                current_stride *= conv_def.stride

            if isinstance(conv_def,Conv):
                end_point = end_point_base
                net = resnet_utils.conv2d_same(net,depth(conv_def.depth),conv_def.kernel,
                                               stride=conv_def.stride,
                                               scope=end_point)
            elif isinstance(conv_def,DepthSepConv):
                end_point = end_point_base + '_depthwise'
                net = separable_conv2d_same(net,conv_def.kernel,
                                            stride=layer_stride,
                                            rate=layer_rate,
                                            scope=end_point)
                end_point = end_point_base + '_pointwise'
                net = slim.conv2d(net,depth(conv_def.depth),[1,1],
                                  stride=1,
                                  scope=end_point)
            else:
                raise ValueError('Unknown convolution type %s for layer %d'%(conv_def.ltype,i))
        return net

def mobilenet_v1_arg_scope(is_training=True,
                           stddev=0.09):
    batch_norm_params = {
        'is_training':False,
        'center':True,
        'scale':True,
        'decay':0.9997,
        'epsilon':0.001,
        'trainable':False,
    }
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(cfg.MOBILENET.WEIGHT_DECAY)
    if cfg.MOBILENET.REGU_DEPTH:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with arg_scope([slim.conv2d,slim.separable_conv2d],
                   trainable=is_training,
                   weights_initializer = weights_init,
                   activation_fn = tf.nn.relu6,             # max(0.6,x)
                   normalizer_fn=slim.batch_norm,
                   padding = 'SAME'):
        with arg_scope([slim.batch_norm],**batch_norm_params):
            with arg_scope([slim.conv2d],weights_regularizer=regularizer):
                with arg_scope([slim.separable_conv2d],
                               weights_regularizer=depthwise_regularizer) as sc:
                    return sc

class mobilenetv1(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16,]
        self._feat_compress = [1. / float(self._feat_stride[0]),]
        self._depth_multiplier = cfg.MOBILENET.DEPTH_MULTIPLIER
        self._scope = 'Mobilenetv1'

    def _image_to_head(self,is_training,reuse=None):
        # 基本瓶颈层
        assert (0<= cfg.MOBILENET.FIXED_LAYERS <= 12)
        net_conv = self._image
        if cfg.MOBILENET.FIXED_LAYERS > 0:
            with arg_scope(mobilenet_v1_arg_scope(is_training=False)):
                net_conv = mobilenet_v1_base(net_conv,
                                             _CONV_DEFS[:cfg.MOBILENET.FIXED_LAYERS],
                                             starting_layer=0,
                                             depth_multiplier=self._depth_multiplier,
                                             reuse=reuse,
                                             scope=self._scope)
        if cfg.MOBILENET.FIXED_LAYERS < 12:
            with arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
                net_conv = mobilenet_v1_base(net_conv,
                                             _CONV_DEFS[cfg.MOBILENET.FIXED_LAYERS:12],
                                             starting_layer=cfg.MOBILENET.FIXED_LAYERS,
                                             depth_multiplier=self._depth_multiplier,
                                             reuse=reuse,
                                             scope=self._scope)
        self._act_summaries.append(net_conv)
        self._layers['head'] = net_conv
        return net_conv

    def _head_to_tail(self,pool5,is_training,reuse=None):
        with arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
            fc7 = mobilenet_v1_base(pool5,
                                    _CONV_DEFS[12:],
                                    starting_layer=12,
                                    depth_multiplier=self._depth_multiplier,
                                    reuse=reuse,
                                    scope=self._scope)
            fc7 = tf.reduce_mean(fc7,axis = [1,2])
        return fc7

    def get_variables_to_restore(self,variables,var_keep_dic):
        variables_to_restore = []
        for v in variables:
            if v.name == (self._scope+'/Conv2d_0/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored in: %s'%v.name)
                variables_to_restore.append(v)
        return variables_to_restore

    def fix_variables(self,sess,pretrained_model):
        print('Fix MobileNet V1 layers..')
        with tf.variable_scope('Fix_MobileNet_V1') as scope:
            with tf.device('/cpu:0'):
                Conv2d_0_rgb = tf.get_variable('Conv2d_0_rgb',
                                               [3,3,3,max(int(32*self._depth_multiplier),8)],
                                               trainable=False)
                restorer_fc = tf.train.Saver({self._scope+'/Conv2d_0/weights':Conv2d_0_rgb})
                restorer_fc.restore(sess,pretrained_model)
                sess.run(tf.assign(self._variables_to_fix[self._scope+'/Conv2d_0/weights:0'],
                                   tf.reverse(Conv2d_0_rgb/(255.0/2.0),[2])))
