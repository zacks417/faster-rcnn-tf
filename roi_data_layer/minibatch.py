from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb,num_classes):
    num_images = len(roidb)
    # scale为600
    random_scale_inds = npr.randint(0,high=len(cfg.TRAIN.SCALES),size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE %num_images==0) ,\
        'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images,cfg.TRAIN.BATCH_SIZE)

    im_blob,im_scales = _get_image_blob(roidb,random_scale_inds)

    blobs = {'data':im_blob}

    assert len(im_scales) ==1,'Single batch only'
    assert len(roidb) == 1,'Single batch only'
    # gt boxes (x1,y1,x2,y2,cls)
    if cfg.TRAIN.USE_ALL_GT:
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        gt_inds = np.where(roidb[0]['gt_classes'] !=0 & np.all(roidb[0]['gt_overlaps'].toarray()>-1.0,axis=1))[0]
    gt_boxes = np.empty((len(gt_inds),5),dtype=np.float32)
    gt_boxes[:,0:4] = roidb[0]['boxes'][gt_inds,:] * im_scales[0]
    gt_boxes[:,4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array([im_blob.shape[1],im_blob.shape[2],im_scales[0]],dtype=np.float32)

    return blobs

def _get_image_blob(roidb,scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:,::-1,:]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im,im_scale = prep_im_for_blob(im,cfg.PIXEL_MEANS,target_size,cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)
    blob = im_list_to_blob(processed_ims)
    return blob,im_scales
