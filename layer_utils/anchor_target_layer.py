from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from model.config import cfg
import numpy as np
import numpy.random as npr
from utils.bbox import bbox_overlaps
from model.bbox_transform import bbox_transform

# 返回rpn生成的标签，目标框，内外权重
def anchor_target_layer(rpn_cls_score,gt_boxes,im_info,_feat_stride,all_anchors,num_anchors):
    A = num_anchors
    total_anchors = all_anchors.shape[0]
    K = total_anchors/num_anchors

    # 允许有少量盒子在边缘上
    _allowed_border = 0
    # 分数图大小 (...,H,W)
    height,width = rpn_cls_score.shape[1:3]

    # 只保留在图片里面的anchors
    inds_inside = np.where(
        (all_anchors[:,0]>= -_allowed_border)&
        (all_anchors[:,1]>= -_allowed_border)&
        (all_anchors[:,2]<im_info[1]+_allowed_border)&  #宽度
        (all_anchors[:,3]<im_info[0]+_allowed_border)   # 高度
    )[0]
    anchors = all_anchors[inds_inside,:]
    # 标签1为正，0为负，-1不用关心,先用-1填充
    labels = np.empty((len(inds_inside),),dtype=np.float32)
    labels.fill(-1)

    # 计算anchors和gt boxes的交叠面积
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors,dtype=np.float),
        np.ascontiguousarray(gt_boxes,dtype=np.float))
    # 对于每个anchor，重叠最大的gt编号
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)),argmax_overlaps]
    # 对于每个gt，重叠最大的anchor编号
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps==gt_max_overlaps)[0]

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps<cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # 对每一个地面框，anchor有最高重叠的置为1
    labels[gt_argmax_overlaps] = 1
    # 超过阈值(0.7)的置为1
    labels[max_overlaps>=cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if  cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        labels[max_overlaps<cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # 如果有太多正标签，再采样
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels==1)[0]
    # 如果超过数量 将超过部分标签置为-1
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds,size=(len(fg_inds)-num_fg),replace=False)
        labels[disable_inds] = -1

    # 如果负标签，同样再采样
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels==1)
    bg_inds = np.where(labels==0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds,size=(len(bg_inds)-num_bg),replace=False)
        labels[disable_inds] = -1
    fg_num = len(np.where(labels==1)[0])
    bg_num = len(np.where(labels==0)[0])
    print('RPN fg:bg = %d : %d'%(fg_num,bg_num))
    bbox_targets = np.zeros((len(inds_inside),4),dtype=np.float32)
    # 这里传入的是对于每个anchor 重叠最大的gtbox,计算anchors和gt之间的回归计算目标
    bbox_targets = _compute_targets(anchors,gt_boxes[argmax_overlaps,:])

    bbox_inside_weights = np.zeros((len(inds_inside),4),dtype=np.float32)
    # 只有正样本才有回归目标
    bbox_inside_weights[labels==1,:] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside),4),dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # 均匀加权
        num_examples = np.sum(labels>=0)
        positive_weights = np.ones((1,4))*1.0/num_examples
        negative_weights = np.ones((1,4))*1.0/num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT>0)&(cfg.TRAIN.RPN_POSITIVE_WEIGHT<1))
        positive_weights = cfg.TRAIN.RPN_POSITIVE_WEIGHT / np.sum(labels==1)
        negative_weights = (1.0-cfg.TRAIN.RPN_POSITIVE_WEIGHT) / np.sum(labels==0)
    bbox_outside_weights[labels==1,:] = positive_weights
    bbox_outside_weights[labels==0,:] = negative_weights

    # 映射到原始anchors集,也就是说原来所有的anchors中，只有刚才的anchors才有回归目标
    labels = _unmap(labels,total_anchors,inds_inside,fill=-1)
    bbox_targets = _unmap(bbox_targets,total_anchors,inds_inside,fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights,total_anchors,inds_inside,fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights,total_anchors,inds_inside,fill=0)
    # 标签
    labels = labels.reshape((1,height,width,A)).transpose(0,3,1,2)
    labels = labels.reshape((1,1,A*height,width))
    rpn_labels = labels
    # 目标框
    bbox_targets = bbox_targets.reshape((1,height,width,A*4))
    rpn_bbox_targets = bbox_targets

    bbox_inside_weights = bbox_inside_weights.reshape((1,height,width,A*4))
    rpn_bbox_inside_weights = bbox_inside_weights

    bbox_outside_weights = bbox_outside_weights.reshape((1,height,width,A*4))
    rpn_bbox_outside_weights = bbox_outside_weights
    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights

def _unmap(data,count,inds,fill=0):
    # 将子集反映射回原始数据集
    if len(data.shape)==1:
        ret = np.empty((count,),dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        # size+data.shape[1:] ret的维度即变为(count,data.shape[1:0])
        ret = np.empty((count,)+data.shape[1:],dtype=np.float32)
        ret.fill(fill)
        ret[inds,:] = data
    return ret

def _compute_targets(ex_rois,gt_rois):
    # 计算一张照片的边界框回归目标
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    return bbox_transform(ex_rois,gt_rois[:,:4]).astype(np.float32,copy=False)



