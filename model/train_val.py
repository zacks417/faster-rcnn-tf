from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

# 训练过程的包装
class SolverWrapper(object):
    def __init__(self,sess,network,imdb,roidb,valroidb,output_dir,tbdir,pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.valroidb = valroidb
        self.output_dir = output_dir
        self.tbdir = tbdir
        self.tbvaldir = tbdir + '_val'
        if not os.path.exists(self.tbvaldir):
            os.makedirs(self.tbvaldir)
        self.pretrained_model = pretrained_model

    def snapshot(self,sess,iter):
        net = self.net
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter)+'.ckpt'
        filename = os.path.join(self.output_dir,filename)
        self.saver.save(sess,filename)
        print('Wrote snapshot to:{:s}'.format(filename))

        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter)+'.pkl'
        nfilename = os.path.join(self.output_dir,nfilename)
        st0 = np.random.get_state()
        cur = self.data_layer._cur
        perm = self.data_layer._perm
        cur_val = self.data_layer_val._cur
        perm_val = self.data_layer_val._perm

        with open(nfilename,'wb') as fid:
            # 序列化对象，并将结果数据流写入到文件对象中
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)
        return filename,nfilename

    def from_snapshot(self,sess,sfile,nfile):
        print('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess,sfile)
        print('Restored')
        with open(nfile,'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm
            self.data_layer_val._cur = cur_val
            self.data_layer_val._perm = perm_val
        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self,file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def construct_graph(self,sess):
        with sess.graph.as_default():
            tf.set_random_seed(cfg.RNG_SEED)
            layers = self.net.create_architecture('TRAIN',self.imdb.num_classes,tag='default',
                                                  anchor_scales = cfg.ANCHOR_SCALES,
                                                  anchor_ratios = cfg.ANCHOR_RATIOS)
            # 定义损失
            loss = layers['total_loss']
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE,trainable=False)
            self.optimizer = tf.train.AdagradOptimizer(lr)
            # self.optimizer = tf.train.MomentumOptimizer(lr,cfg.TRAIN.MOMENTUM)
            # 根据损失计算梯度
            gvs = self.optimizer.compute_gradients(loss)
            # 如果设置了偏置上梯度翻倍
            if cfg.TRAIN.DOUBLE_BIAS:
                final_gvs = []
                with tf.variable_scope('Gradient_Mult') as scope:
                    for grad,var in gvs:
                        scale = 1.
                        if cfg.TRAIN.DOUBLE_BIAS and '/biases:' in var.name:
                            scale *= 2.
                        # allclose 检测是否相同
                        if not np.allclose(scale,1.0):
                            grad = tf.multiply(grad,scale)
                        final_gvs.append((grad,var))
                train_op = self.optimizer.apply_gradients(final_gvs)
            else:
                train_op = self.optimizer.apply_gradients(gvs)

            self.saver = tf.train.Saver(max_to_keep=100000)
            self.writer = tf.summary.FileWriter(self.tbdir,sess.graph)
            self.valwriter = tf.summary.FileWriter(self.tbvaldir)

        return lr,train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir,cfg.TRAIN.SNAPSHOT_PREFIX+'_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self,sess):
        np_paths = []
        ss_paths = []
        # 直接从加载ImageNet权重开始
        variables = tf.global_variables()
        # with open('res101.txt','w') as f:
        #     for variable in variables:
        #         f.write(str(variable))
        #         f.write('\n')
        # 首先初始化所有变量
        sess.run(tf.variables_initializer(variables, name='init'))
        if self.pretrained_model is not None:
            print('Loading initial model weights from {:s}'.format(self.pretrained_model))
                #     var_keep_dic = self.get_variables_in_checkpoint_file(
                #'D:\Faster_R-CNN\output/res_5/res101_faster_rcnn_iter_15.ckpt')
            var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
            variables_to_restore = self.net.get_variables_to_restore(variables,var_keep_dic)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess,self.pretrained_model)
            print('Loaded.')
        # if self.pretrained_model is not None:
        #     print('variables to restore')
        #     # variable = tf.contrib.framework.get_variables_to_restore()
        #     # print(variable)
        #     # exit()
        #     names =['block1','block2','blcok3','block4','conv1']
        #     # variables_to_restore = [v for v in variable if v.name.split('/')[1] in names]
        #     variables_to_restore = tf.trainable_variables()
        #     variables_to_restore = [v for v in variables_to_restore if v.name.split('/')[1] in names]
        #     restorer = tf.train.Saver(variables_to_restore)
        #     restorer.restore(sess,self.pretrained_model)
        #     print('Loaded')
        # if self.pretrained_model is not None:
        #     print('Loading part of vars from {:s}'.format(self.pretrained_model))
        #     restorer = tf.train.Saver(variables[1])
        #     restorer.restore(sess,self.pretrained_model)
        #     print('Loaded!')
        #需要固定变量，RGB能变为BGR
            self.net.fix_variables(sess,self.pretrained_model)
        print('Fixed.')
        last_snapshot_iter = 0
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = list(cfg.TRAIN.STEPSIZE)

        return rate,last_snapshot_iter,stepsizes,np_paths,ss_paths

    def restore(self,sess,sfile,nfile):
        # 获取最新的快照并恢复
        np_paths = [nfile]
        ss_paths = [sfile]
        # 从快照中恢复模型
        last_snapshot_iter = self.from_snapshot(sess,sfile,nfile)
        # 设定学习率
        rate = cfg.TRAIN.LEARNING_RATE
        stepsizes = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            if last_snapshot_iter > stepsize:
                rate *= cfg.TRAIN.GAMMA
            else:
                stepsizes.append(stepsize)
        return rate,last_snapshot_iter,stepsizes,np_paths,ss_paths

    # 移除快照
    def remove_snapshot(self,np_paths,ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)
        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile+'.data-00000-of-00001'))
                os.remove(str(sfile+'.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def train_model(self,sess,max_iters):
        # 同时对训练和验证集创建数据层
        self.data_layer = RoIDataLayer(self.roidb,self.imdb.num_classes)
        self.data_layer_val = RoIDataLayer(self.valroidb,self.imdb.num_classes,random=True)
        # 执行计算图
        lr, train_op = self.construct_graph(sess)
        # 寻找前一个快照
        lsf, nfiles, sfiles = self.find_previous()
        # 初始化变量 或者从最近快照恢复
        if lsf == 0:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize(sess)
        else:
            rate, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(sess,str(sfiles[-1]),str(nfiles[-1]))
        timer = Timer()
        iter = last_snapshot_iter + 1
        last_summary_time = time.time()
        # 保证列表非空
        stepsizes.append(max_iters)
        stepsizes.reverse()
        # 下一步steo大小 由队列出栈
        next_stepsize = stepsizes.pop()
        while iter < max_iters+1:
            # 学习率
            if iter == next_stepsize+1:
                # 在减少学习率之前添加快照
                self.snapshot(sess,iter)
                rate *= cfg.TRAIN.GAMMA
                sess.run(tf.assign(lr,rate))
                next_stepsize = stepsizes.pop()
            # 单次开始计时
            timer.tic()

            # 获取一个minibatch大小训练数据
            blobs = self.data_layer.forward()
            now = time.time()
            if iter ==1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
                # 带着summary计算图
                rpn_loss_cls,rpn_loss_box,loss_cls,loss_box,total_loss,summary = \
                    self.net.train_step_with_summary(sess,blobs,train_op)
                self.writer.add_summary(summary,float(iter))
                # 在验证集上检测summary
                blobs_val = self.data_layer_val.forward()
                summary_val = self.net.get_summary(sess,blobs_val)
                self.valwriter.add_summary(summary_val,float(iter))
                last_summary_time = now
            else:
                # 不带summary计算图
                rpn_loss_cls,rpn_loss_box,loss_cls,loss_box,total_loss = \
                    self.net.train_step(sess,blobs,train_op)
            # 结束本次计时a
            timer.toc()

            # 显示训练信息,10iters
            if iter % cfg.TRAIN.DISPLAY == 0:
                print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                      '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                      (iter, max_iters, total_loss, rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, lr.eval()))
                print('speed: {:.3f}s / iter'.format(timer.average_time))

            # 快照，500
            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                ss_path, np_path = self.snapshot(sess,iter)
                np_paths.append(np_path)
                ss_paths.append(ss_path)
                # 如果超过快照保存数  移除旧的
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths,ss_paths)
            iter += 1
        if last_snapshot_iter != iter -1:
            self.snapshot(sess,iter-1)
        self.writer.close()
        self.valwriter.close()

def get_training_roidb(imdb):
    # 返回训练用的region of interest database
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')
    print('Preparing training data...')
    # 扩充roidb保存信息，增加保存路径，尺寸，最大重叠以及对应的类别
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb

def filter_roidb(roidb):
    # 移除没有用的rois
    def is_valid(entry):
        # 有用的图片 至少有一个前景，一个背景
        overlaps = entry['max_overlaps']
        # 找到有足够重叠面积的盒子
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # 选择背景roi
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI)&(overlaps>=cfg.TRAIN.BG_THRESH_LO))[0]
        valid = len(fg_inds) >0 or len(bg_inds) > 0
        return valid
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num-num_after,num,num_after))
    return filtered_roidb

def train_net(network,imdb,roidb,valroidb,output_dir,tb_dir,
              pretrained_model = None,
              max_iters = 40000):
    # 训练一个faster R-CNN 网络
    roidb = filter_roidb(roidb)
    valroidb = filter_roidb(valroidb)
    # 当运行设备不满足要求时，会自动分配GPU 或者CPU
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # 自动慢慢达到最大GPU内存
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess,network,imdb,roidb,valroidb,output_dir,tb_dir,pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess,max_iters)
        print('done solving')

