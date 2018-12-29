import numpy as np

def nms(dets,threshold):
    # dets 为x1 y1 x2 y2 score
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]
    # 计算每个框的面积
    areas = (x2-x1+1)*(y2-y1+1)
    # 将得分从高到底排序
    score_order = scores.argsort()[::-1]

    keep = []
    while score_order.size > 0:
        # 每次选取当前得分最高的索引，先加入到保留队列中
        i = score_order[0]
        keep.append(i)
        # 计算该框与其他框的重叠面积
        xx1 = np.maximum(x1[i],x1[score_order[1:]])
        yy1 = np.maximum(y1[i],y1[score_order[1:]])
        xx2 = np.minimum(x2[i],x2[score_order[1:]])
        yy2 = np.minimum(y2[i],y2[score_order[1:]])
        # 计算重叠面积前要先判断高宽是否为负数，只能为非负
        w = np.maximum(0.0,xx2-xx1+1)
        h = np.maximum(0.0,yy2-yy1+1)
        inter = w*h
        iou = inter / (areas[i]+areas[score_order[1:]]-inter)

        inds = np.where(iou <= threshold)[0]
        # 这里要加1是因为iou相比原来是少了一个数的
        score_order = score_order[inds+1]
    return keep
