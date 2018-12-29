from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.test import test_net
from model.config import cfg,cfg_from_file,cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time,os,sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    parser.add_argument('--cfg',dest='cfg_file',
                        help = 'optional config file',default=None,type=str)
    parser.add_argument('--model',dest='model',
                        help='model to test',
                        default='D:\Faster_R-CNN\output/voc_2007_trainval\default/res101_faster_rcnn_iter_300000.ckpt',
                        type=str)
    parser.add_argument('--imdb',dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test',type=str)
    parser.add_argument('--comp',dest='comp_mode',help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets',dest='max_per_image',
                        help = 'max number of detections per image',
                        default=100,type=int)
    parser.add_argument('--tag',dest='tag',
                        help='tag of the model',
                        default='',type=str)
    parser.add_argument('--net',dest='net',
                        help = 'vgg16, res50, res101, res152, mobile',
                        default='res101',type=str)
    parser.add_argument('--set',dest='set_cfgs',
                        help='set config keys',default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]
    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    sess = tf.Session(config=tfconfig)

    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    net.create_architecture('TEST',imdb.num_classes,tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)
    if args.model:
        print('Loading model check point from {:s}'.format(args.model))
        saver = tf.train.Saver()
        saver.restore(sess,args.model)
        print('Loaded')
    else:
        print('Loading initial weights from {:s}'.format(args.weight))
        sess.run(tf.global_variables_initializer())
        print('Loaded')

    test_net(sess,net,imdb,filename,max_per_image=args.max_per_image)

    sess.close()








