以**VGG16**为backbone来讲解faster-rcnn整体流程

![](https://i.imgur.com/QLUbNP7.png)

以vgg网络输入为**600x800x3**为例，网络的**conv5_3**为rpn网络的输入，就是上图中的feature Map, 大小为**1x38x50x512**

**RPN部分：**

下图为rpn的结构图,输入即为conv5_3，输出rois（128x84）128个感兴趣区域，84为21x4，即为每个类别都预测bbox。

![](https://i.imgur.com/VaPHjem.png)

conv5-3首先通过3x3 same卷积降维，记为rpn-conv，维度变成1x38x50x512，接下去是两个分支，都用的1x1卷积，数量分别为anchorsx2（前景后景）,x4（两个点坐标）。

下面分支**rpn_bbox_pred**(1x38x50x36)为预测框的坐标和anchor的offset。

上面rpn_cls_score(1x38x50x18，这里最后为18，9x2，是包含了背景的，一会计算标签hloss会去掉背景)为分类得分图，最上面分支中通过加入地面真实框、图像高宽信息、网络总步长、总anchors数量(9)通过anchor-target-layer层生成rpn-label(1x1x38x50),这个函数在代码文件./layer_utils/anchor-target-layer中，该函数同时还生成rpn-bbox-targets(anchor和gt-box之间的offset回归目标)以及相对应的权重rpn-bbox-inside-weights,rpn-bbox-outside-weights，等会用于计算rpn-loss-box和rpn-cross-entropy。

然后是rpn-cls-score下面这个分支做一次reshape为rpn-cls-score-reshape（维度1x38x(50x9)x2，这个去除掉-1的标签之后与rpn-label求交叉softmax损失），便于做softmax二分类,接着reshape成[38x50x9,2]在第二个维度2上进行argmax求索引左为类别预测结果rpn-cls-pred维度为(38x50x9,)；

将rpn-cls-score-reshape在最后一个维度(2)上做softmax二分类为rpn-cls-prob-reshape（维度为1x38x(50x9)x2）然后再把这个9 reshape回去为rpn-cls-prob（维度1x38x50x18）

此时加入地面真实框、图像高宽、网络步长、总anchor数(9)、之前预测的rpn-bbox-pred通过一层proposal-layer生成rois和对应的rois-scores（nms之前先按得分排序选取前12000个，nms后剩余2000个proposal）（2000x5，5是序号+两个坐标）。


然后再加入地面真实框和类别数(20+1背景也算一类而且是序号0)经过proposal-target-layer生成rois（128x84，21x4每个类别都有坐标）,同时还会生成_proposal_targets字典，存放rois、labels、bbox_targets、bbox_inside_weights、bbox_outside_weights，等会用来计算RCNN的损失反过来进行修正。到此，rpn网络已经产生输出建议区域。

下面先列一下**rpn网络的两个损失**，之前都有提到过：

**RPN分类损失：**

rpn-cross-entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn-cls-score,labels=rpn-label))。

**RPN回归损失：**

这里是计算预测框的坐标和anchors的offset 和 anchors和地面真实框的offset，两个offset的差值作为回归目标,用smooth-l1损失计算
rpn-loss-box = self.-smooth-l1-loss(rpn-bbox-pred,rpn-bbox-targets,rpn-bbox-inside-weights,rpn-bbox-outside-weights,sigma=sigma_rpn,dim=[1,2,3])

**ROIPooling**部分，如下图所示，前面rpn生成的rois感兴趣区域中有两个坐标点，将它映射到conv5-3的特征图上即为感兴趣区域，对这一部分进行池化，-crop-pool-layer函数在network.py中这版代码没有实现ROI Pooling layer 而是把ROI对应的特征图resize成相同尺寸（14x14）后再进行 max pooling（之后是7x7）生成pool5。 （后面tf直接有ROIAlign中采用的双线性插值的裁剪图片方法tf.image.crop-and-resize(images, boxes, batch_inds,[pooled_height, pooled_width], method='bilinear', name='Crop')）

![](https://i.imgur.com/mtt6Jhy.png)


**RCNN分类和回归：**

输入是之前的pool5（128x7x7x512），经过两层全连接层（后接relu）然后分别用全连接层生成目标框预测结果bbox_pred（128x4）、全连接层后接softmax生成类别预测结果cls_prob(128x21)。结构如下图所示

![](https://i.imgur.com/Gp4P9jo.png)

下面介绍一下RCNN的损失，也是两类

**分类损失**

这里的输入是最后全连接层的输出，没有经过softmax的（对应tf.nn.sparse_softmax_cross_entropy_with_logits这个函数）,标签是RPN网络最后proposal_target_layer生成的label
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score,labels=label))


**回归损失**

输入一个为最后全连接层的输出，另一个为RPN最后的bbox输出，用来对RPN输出的bbox进行修正。
loss_box = self._smooth_l1_loss(bbox_pred,bbox_targets,bbox_inside_weights,bbox_outside_weights)


