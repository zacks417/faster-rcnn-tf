from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
from layer_utils.snippets import generate_anchors_pre,generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer,proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer,proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes
from model.config import cfg

class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # 把平均值加回来
        image = self._image + cfg.PIXEL_MEANS
        # BGR dao RGB
        resized = tf.image.resize_bilinear(image,tf.to_int32(self._im_info[:2]/self._im_info[2]))
        self._gt_image = tf.reverse(resized,axis=[-1])

    def _add_gt_image_summary(self):
        # 使用自定的可视化函数可视化框
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image,self._gt_boxes,self._im_info],
                           tf.float32,name='gt_boxes')
        return tf.summary.image('GROUND_TRUTH',image)

    def _add_act_summary(self,tensor):
        tf.summary.histogram('ACT/'+tensor.op.name+'/activations',tensor)
        tf.summary.scalar('ACT/'+tensor.op.name+'/zero_fraction',tf.nn.zero_fraction(tensor))

    def _add_score_summary(self,key,tensor):
        tf.summary.histogram('SCORE/'+tensor.op.name+'/'+key+'/scores',tensor)

    def _add_train_summary(self,var):
        tf.summary.histogram('TRAIN/'+var.op.name,var)

    def _reshape_layer(self,bottom,num_dim,name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # 将通道改为 caffe的形式
            to_caffe = tf.transpose(bottom,[0,3,1,2])
            reshaped = tf.reshape(to_caffe,tf.concat(axis=0,values=[[1,num_dim,-1],[input_shape[2]]]))
            to_tf = tf.transpose(reshaped,[0,2,3,1])
            return to_tf

    def _softmax_layer(self,bottom,name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom,[-1,input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped,name=name)
            return tf.reshape(reshaped_score,input_shape)
        return tf.nn.softmax(bottom,name=name)

    def _proposal_top_layer(self,rpn_cls_prob,rpn_bbox_pred,name):
        # 不用NMS 选择前5000个
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois,rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois,rpn_scores = tf.py_func(proposal_top_layer,
                                             [rpn_cls_prob,rpn_bbox_pred,self._im_info,
                                              self._feat_stride,self._anchors,self._num_anchors],
                                             [tf.float32,tf.float32],name='proposal_top')
            rois.set_shape([cfg.TEST.RPN_TOP_N,5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N,1])
        return rois,rpn_scores

    def _proposal_layer(self,rpn_cls_prob,rpn_bbox_pred,name):
        # 生成rois，roi_scores
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois,rpn_scores = proposal_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._mode,
                    self._feat_stride,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois,rpn_scores = tf.py_func(proposal_layer,
                                             [rpn_cls_prob,rpn_bbox_pred,self._im_info,self._mode,
                                              self._feat_stride,self._anchors,self._num_anchors],
                                             [tf.float32,tf.float32],name='proposal')
            rois.set_shape([None,5])
            rpn_scores.set_shape([None,1])
        return rois,rpn_scores

    def _roi_pool_layer(self,bottom,rois,name):
        with tf.variable_scope(name) as scope:
            return tf.image.roi_pooling(bottom,rois,
                                        pooled_height = cfg.POOLING_SIZE,
                                        pooled_width = cfg.POOLING_SIZE,
                                        spatial_scale = 1./16.)[0]
    # 这版代码没有实现ROI Pooling layer 而是把ROI对应的特征图resize成相同尺寸后再进行 max pooling
    def _crop_pool_layer(self,bottom,rois,name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois,[0,0],[-1,1],name='batch_id'),[1])
            # 边界框标准化,因为tf.image.crop_and_resize第二个参数传入的是归一化坐标,
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1])-1.)*np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2])-1.)*np.float32(self._feat_stride[0])
            x1 = tf.slice(rois,[0,1],[-1,1],name='x1') / width
            y1 = tf.slice(rois,[0,2],[-1,1],name='y1') / height
            x2 = tf.slice(rois,[0,3],[-1,1],name='x2') / width
            y2 = tf.slice(rois,[0,4],[-1,1],name='y2') / height
            # 不会后向传播到rois
            bboxes = tf.stop_gradient(tf.concat([y1,x1,y2,x2],axis=1))
            # 7*2=14
            pre_pool_size = cfg.POOLING_SIZE * 2
            crops = tf.image.crop_and_resize(bottom,bboxes,tf.to_int32(batch_ids),[pre_pool_size,pre_pool_size],name='crops')
            # 默认maxpooling步长为2
        return slim.max_pool2d(crops,[2,2],padding='SAME')

    def _dropout_layer(self,bottom,name,ratio=0.5):
        return tf.nn.dropout(bottom,ratio,name=name)

    def _anchor_target_layer(self,rpn_cls_score,name):
        # 这一层一方面生成了rpn_labels，还生成的计算rpn回归损失用的rpn_bbox_targets(an和gt的offset)还有内外权重
        with tf.variable_scope(name) as scope:
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score,self._gt_boxes,self._im_info,self._feat_stride,self._anchors,self._num_anchors],
                [tf.float32,tf.float32,tf.float32,tf.float32],
                name='anchor_target'
            )
            rpn_labels.set_shape([1,1,None,None])
            rpn_bbox_targets.set_shape([1,None,None,self._num_anchors*4])
            rpn_bbox_inside_weights.set_shape([1,None,None,self._num_anchors*4])
            rpn_bbox_outside_weights.set_shape([1,None,None,self._num_anchors*4])

            rpn_labels = tf.to_int32(rpn_labels,name='to_int32')
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)
        return rpn_labels

    def _proposal_target_layer(self,rois,roi_scores,name):
        # 为上面proposal_layer生成的rois进一步分配一些值,有RCNN里回归和分类损失用到的labels，bbox_targets和权重
        with tf.variable_scope(name) as scope:
            rois,roi_scores,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois,roi_scores,self._gt_boxes,self._num_classes],
                [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32],
                name='proposal_target'
            )
            rois.set_shape([cfg.TRAIN.BATCH_SIZE,5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE,1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE,self._num_classes*4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,self._num_classes*4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,self._num_classes*4])

            self._proposal_targets['rois'] = rois
            # labels = tf.to_int32(labels,name='to_int32')
            self._proposal_targets['labels'] = tf.to_int32(labels,name='to_int32')
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)
        return rois,roi_scores

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_'+self._tag) as scope:
            # ceil 向上取整
            height = tf.to_int32(tf.ceil(self._im_info[0]/np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1]/np.float32(self._feat_stride[0])))
            if cfg.USE_E2E_TF:
                anchors ,anchor_length = generate_anchors_pre_tf(
                    height,
                    width,
                    self._feat_stride,
                    self._anchor_scales,
                    self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height,width,
                                                     self._feat_stride,self._anchor_scales,self._anchor_ratios],
                                                    [tf.float32,tf.int32],name='generate_anchors')
            anchors.set_shape([None,4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self,is_training=True):
        # 选择初始化方式
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0,stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0,stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0,stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0,stddev=0.001)

        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope,self._scope):
            # 生成anchors
            self._anchor_component()
            # 区域建议网络
            rois = self._region_proposal(net_conv,is_training,initializer)
            if cfg.POOLING_MODE == 'crop':
                pool5 = self._crop_pool_layer(net_conv,rois,'pool5')

            else:
                raise  NotImplementedError
        fc7 = self._head_to_tail(pool5,is_training)
        with tf.variable_scope(self._scope,self._scope):
            # 区域分类
            cls_prob, bbox_pred = self._region_classification(fc7,is_training,
                                                              initializer,initializer_bbox)
        self._score_summaries.update(self._predictions)

        return rois,cls_prob,bbox_pred

    def _smooth_l1_loss(self,bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights,sigma=1.0,dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        # 这里判断g(x) 中x范围
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff,1./sigma_2)))
        in_loss_box = tf.pow(in_box_diff,2)*(sigma_2/2.)*smoothL1_sign + (abs_in_box_diff-(0.5/sigma_2))*(1.-smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box,axis=dim))
        return loss_box

    def _add_losses(self,sigma_rpn=3.0):
        with tf.variable_scope('LOSS_'+self._tag) as scope:
            # RPN ,分类损失
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'],[-1,2])
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'],[-1])
            # 剔除标签为-1的无关项
            rpn_select = tf.where(tf.not_equal(rpn_label,-1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,rpn_select),[-1,2])
            rpn_label = tf.reshape(tf.gather(rpn_label,rpn_select),[-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,labels=rpn_label))

            # RPN, 边界框损失
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred,rpn_bbox_targets,rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights,sigma=sigma_rpn,dim=[1,2,3])

            #  RCNN,分类损失
            cls_score = self._predictions['cls_score']
            label = tf.reshape(self._proposal_targets['labels'],[-1])
            cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score,labels=label))

            # RCNN, 边界框损失
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            # self._losses['total_loss'] = loss
            # 加上正则化损失
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(),'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)
        return loss

    def _region_proposal(self,net_conv,is_training,initializer):
        # 输出通道512
        rpn = slim.conv2d(net_conv,cfg.RPN_CHANNELS,[3,3],trainable=is_training,weights_initializer=initializer,
                          scope='rpn_conv/3x3')
        self._act_summaries.append(rpn)
        # shape = (1,x,x,18)   num_anchors为9
        rpn_cls_score = slim.conv2d(rpn,self._num_anchors*2,[1,1],trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID',activation_fn=None,scope='rpn_cls_score')
        # 进行reshape,原来为[1,2*9,H,W],变为[1,2,9*H,W] 方便做softmax二分类
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score,2,'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape,'rpn_cls_prob_reshape')
        # shape = (?,2)
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape,[-1,2]),axis=1,name='rpn_cls_pred')
        # shape = (1,?,?,18)
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape,self._num_anchors*2,'rpn_cls_prob')
        # shape = (1,?,?,36)
        rpn_bbox_pred = slim.conv2d(rpn,self._num_anchors*4,[1,1],trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID',activation_fn=None,scope='rpn_bbox_pred')
        if is_training:
            rois,roi_scores = self._proposal_layer(rpn_cls_prob,rpn_bbox_pred,'rois')
            rpn_labels = self._anchor_target_layer(rpn_cls_score,'anchor')
            # 尝试对计算图活动确定性顺序，以获得可重复性
            with tf.control_dependencies([rpn_labels]):
                # 为上一步得到的proposal分配所属物体类别，并得到proposal和gt_bbox的坐标位置间的差别
                rois,_ = self._proposal_target_layer(rois,roi_scores,'rpn_rois')
        else:
            if cfg.TEST.MODE=='nms':
                rois,_ = self._proposal_layer(rpn_cls_prob,rpn_bbox_pred,'rois')
            elif cfg.TEST.MODE=='top':
                rois,_ = self._proposal_top_layer(rpn_cls_prob,rpn_bbox_pred,'rois')
            else:
                raise NotImplementedError

        self._predictions['rpn_cls_score'] = rpn_cls_score
        self._predictions['rpn_cls_score_reshape'] = rpn_cls_score_reshape
        self._predictions['rpn_cls_prob'] = rpn_cls_prob
        self._predictions['rpn_cls_pred'] = rpn_cls_pred
        self._predictions['rpn_bbox_pred'] = rpn_bbox_pred
        self._predictions['rois'] = rois

        return rois

    def _region_classification(self,fc7,is_training,initializer,initializer_bbox):
        cls_score = slim.fully_connected(fc7,self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None,scope='cls_score')
        cls_prob = self._softmax_layer(cls_score,'cls_prob')
        cls_pred = tf.argmax(cls_score,axis=1,name='cls_pred')
        bbox_pred = slim.fully_connected(fc7,self._num_classes*4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None,scope='bbox_pred')
        self._predictions['cls_score'] = cls_score
        self._predictions['cls_pred'] = cls_pred
        self._predictions['cls_prob'] = cls_prob
        self._predictions['bbox_pred'] = bbox_pred

        return cls_prob,bbox_pred

    def _image_to_head(self,is_training,reuse=None):
        raise NotImplementedError
    # 在子类中实现
    def _head_to_tail(self,pool5,is_training,reuse=None):
        raise NotImplementedError

    def create_architecture(self,mode,num_classes,tag=None,
                            anchor_scales=(8,16,32),anchor_ratios=(0.5,1,2)):
        self._image = tf.placeholder(tf.float32,shape=[1,None,None,3])
        self._im_info = tf.placeholder(tf.float32,shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32,shape=[None,5])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        # 这里为3*3=9
        self._num_anchors = self._num_scales * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # 一些正则化
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        with arg_scope([slim.conv2d,slim.conv2d_in_plane,
                        slim.conv2d_transpose,slim.separable_conv2d,slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer = biases_regularizer,
                       biases_initializer = tf.constant_initializer(0.0)):
            rois,cls_prob,bbox_pred = self._build_network(training)

        layers_to_output = {'rois':rois}

        # 将可训练变量全部添加进去
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),(self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS),(self._num_classes))
            self._predictions['bbox_pred'] *=stds
            self._predictions['bbox_pred'] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device('/cpu:0'):
                val_summaries.append(self._add_gt_image_summary())
                for key,var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key,var))
                for key,var in self._score_summaries.items():
                    self._add_score_summary(key,var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self,variables,var_keep_dic):
        raise NotImplementedError

    def fix_variables(self,sess,pretrained_model):
        raise NotImplementedError

    # 提取头部特征图，对于vgg16来说是 conv5_3, 只在测试的时候
    def extract_head(self,sess,image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers['head'],feed_dict=feed_dict)
        return feat

    # 只在测试的时候用
    def test_image(self,sess,image,im_info):
        feed_dict = {self._image:image,
                     self._im_info:im_info}
        cls_score,cls_prob,bbox_pred,rois = sess.run([self._predictions['cls_score'],
                                                      self._predictions['cls_prob'],
                                                      self._predictions['bbox_pred'],
                                                      self._predictions['rois']],
                                                     feed_dict=feed_dict)
        return cls_score,cls_prob,bbox_pred,rois

    def get_summary(self,sess,blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val,feed_dict=feed_dict)
        return summary

    def train_step(self,sess,blobs,train_op):
        feed_dict = {self._image:blobs['data'],self._im_info:blobs['im_info'],
                     self._gt_boxes:blobs['gt_boxes']}
        rpn_loss_cls,rpn_loss_box,loss_cls,loss_box,loss,_ = sess.run([self._losses['rpn_cross_entropy'],
                                                                      self._losses['rpn_loss_box'],
                                                                      self._losses['cross_entropy'],
                                                                      self._losses['loss_box'],
                                                                      self._losses['total_loss'],
                                                                      train_op],
                                                                     feed_dict=feed_dict)
        return rpn_loss_cls,rpn_loss_box,loss_cls,loss_box,loss

    def train_step_with_summary(self,sess,blobs,train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary,_ = sess.run([self._losses['rpn_cross_entropy'],
                                                                            self._losses['rpn_loss_box'],
                                                                            self._losses['cross_entropy'],
                                                                            self._losses['loss_box'],
                                                                            self._losses['total_loss'],
                                                                            self._summary_op,
                                                                            train_op],
                                                                           feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self,sess,blobs,train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op],feed_dict=feed_dict)

