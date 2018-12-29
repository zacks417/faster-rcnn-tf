from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from model.config import cfg
from model.test import im_detect
from nms.nms import nms
from tensorflow.python.platform import gfile
from tensorflow.python import pywrap_tensorflow
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os,cv2
import argparse
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im,class_name,dets,thresh=0.2):
    inds = np.where(dets[:,-1]>=thresh)[0]

    if len(inds) == 0:
        return
    im = im[:,:,(2,1,0)]
    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(im,aspect='equal')
    for i in inds:
        bbox = dets[i,:4]
        score = dets[i,-1]
        ax.add_patch(plt.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],fill = False,
                                   edgecolor='red',linewidth=3.5))
        ax.text(bbox[0],bbox[1]-2,'{:s} {:.3f}'.format(class_name,score),
                bbox=dict(facecolor='blue',alpha=0.5),
                fontsize=14,color='white')
    ax.set_title(('{} detections with p({}|box) >= {:.1f}').format(class_name,class_name,thresh),
                 fontsize = 14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess,net,image_dir,image_name):
    im_file = os.path.join(image_dir,image_name)
    im = cv2.imread(im_file)

    timer = Timer()
    timer.tic()
    scores,boxes = im_detect(sess,net,im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time,boxes.shape[0]))
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind,cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:,4*cls_ind:4*(cls_ind+1)]
        cls_scores = scores[:,cls_ind]
        dets = np.hstack((cls_boxes,cls_scores[:,np.newaxis])).astype(np.float32)
        keep = nms(dets,NMS_THRESH)
        dets = dets[keep,:]
        vis_detections(im,cls,dets,thresh=CONF_THRESH)


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net',dest='demo_net',help='Network to use [vgg16 res101]',
                        default='res101')
    parser.add_argument('--dataset',dest='dataset',default='image')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    cfg.TEST.HAS_RPN = True
    args = parse_args()
    demonet = args.demo_net
    image_dir = 'image'
    dataset = args.dataset
    tfmodel = 'D:\Faster_R-CNN\output/voc_2007_trainval\default/res101_faster_rcnn_iter_300000.ckpt'
    # ckptfile = 'D:\Faster_R-CNN\output\default/vgg_faster_rcnn_iter_6500.ckpt'
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    #加载网络
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else :
        raise NotImplementedError
    net.create_architecture('TEST',21,tag='default',anchor_scales=[8,16,32])
    # reader = pywrap_tensorflow.NewCheckpointReader(ckptfile)
    # variables_to_restore = reader.get_variable_to_shape_map()
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    # im_names  = ['000001.jpg']
    im_names = ['000001.jpg','000010.jpg','000014.jpg','000018.jpg','000020.jpg','000023.jpg',
                 '000024.jpg','000028.jpg','000030.jpg','000035.jpg','000036.jpg']
    for im_name in im_names:
        print('---------------------')
        print(im_name)
        demo(sess,net,image_dir,im_name)
    plt.show()
