from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from model.config import cfg
from model.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps


def proposal_target_layer(rpn_rois,rpn_scores,gt_boxes,_num_classes):
    # 将对象检测建议分配给地面真实框，生成建议分类标签(128)和边界框回归目标(128 x 84)
    # 建议roi (0,x1,y1,x2,y2)
    all_rois = rpn_rois
    all_scores = rpn_scores

    # 在候选rois里包括地面真实框
    if cfg.TRAIN.USE_GT :
        zeros = np.zeros((gt_boxes.shape[0],1),dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois,np.hstack((zeros,gt_boxes[:,:-1]))))
        all_scores = np.vstack((all_scores,zeros))

    num_images = 1
    # 每张图片多少个roi,多少个前景roi
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # 用分类标签和边界框回归来对rois进行采样
    labels, rois, roi_scores, bbox_targets,bbox_inside_weights = _sample_rois(
        all_rois,all_scores,gt_boxes,fg_rois_per_image,rois_per_image,_num_classes)

    rois = rois.reshape(-1,5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1,1)
    bbox_targets = bbox_targets.reshape(-1,_num_classes*4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1,_num_classes*4)
    bbox_outside_weights = np.array(bbox_inside_weights>0).astype(np.float32)

    return rois,roi_scores,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

def _get_bbox_regression_labels(bbox_target_data,num_classes):
    # n x (class,tx,ty,tw,th)
    # 返回目标框 n x 4k,
    clss = bbox_target_data[:,0]
    bbox_targets = np.zeros((clss.size,4*num_classes),dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape,dtype=np.float32)
    # 只有类别大于0的才有非零目标，才有权重
    inds = np.where(clss>0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind,start:end] = bbox_target_data[ind,1:]
        bbox_inside_weights[ind,start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets,bbox_inside_weights

def _compute_targets(ex_rois,gt_rois,labels):
    # 计算图像的边界框回归目标
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois,gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets-np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))/np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack((labels[:,np.newaxis],targets)).astype(np.float32,copy=False)

def _sample_rois(all_rois,all_scores,gt_boxes,fg_rois_per_image,rois_per_image,num_classes):
    # overlaps (rois x gt_boxes)
    overlaps = bbox_overlaps(np.ascontiguousarray(all_rois[:,1:5],dtype=np.float),
                             np.ascontiguousarray(gt_boxes[:,:4],dtype=np.float))
    # 对于每个anchor，重叠最大的gt编号
    gt_assignment = overlaps.argmax(axis = 1)
    max_overlaps = overlaps.max(axis=1)
    # 重叠最大的gt的标签
    labels = gt_boxes[gt_assignment,4]

    # 选择前景rois 大于前景阈值部分
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # 背景rois在背景阈值之间
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI)&(max_overlaps>=cfg.TRAIN.BG_THRESH_LO))[0]
    # bg_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH_HI)[0]
    # 确保对固定数量的区域进行采样
    if fg_inds.size > 0 and bg_inds.size >0:
        fg_rois_per_image = min(fg_rois_per_image,fg_inds.size)
        fg_inds = npr.choice(fg_inds,size=int(fg_rois_per_image),replace=False)
        # 每张图片的背景roi数为总roi-前景数
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        # 如果背景数量少于每张图片背景数，replace置为True，即可以对同一元素反复选取
        to_replace = bg_inds.size < bg_rois_per_image
        bg_inds = npr.choice(bg_inds,size=int(bg_rois_per_image),replace=to_replace)
    elif fg_inds.size>0:
        # 没有背景roi,如果前景少于总的 允许重复
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds,size=int(rois_per_image),replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size>0:
        # 没有前景，如果背景少于总的，允许重复
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds,size=int(rois_per_image),replace=to_replace)
        fg_rois_per_image = 0
    else:
        # 否则 在线调试代码
        print(1)
        import pdb
        pdb.set_trace()

    # 刚才选择的前景和背景序号
    keep_inds = np.append(fg_inds,bg_inds)
    labels = labels[keep_inds]
    # 将背景标签置为0
    labels[int(fg_rois_per_image):] =0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    # 返回目标数据框，标签加四个回归数据目标tx ty tw th
    bbox_target_data = _compute_targets(rois[:,1:5],gt_boxes[gt_assignment[keep_inds],:4],labels)
    # 返回有类别的目标框以及内部权重
    bbox_targets,bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data,num_classes)
    return labels,rois,roi_scores,bbox_targets,bbox_inside_weights
