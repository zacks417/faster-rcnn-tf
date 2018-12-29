import numpy as np

def bbox_overlaps(boxes,query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N,K),dtype=np.float32)
    for k in range(K):
        box_area = (query_boxes[k,2]-query_boxes[k,0]+1)*(query_boxes[k,3]-query_boxes[k,1]+1)
        for n in range(N):
            # 计算重叠部分
            inter_w = min(boxes[n,2],query_boxes[k,2])-max(boxes[n,0],query_boxes[k,0])+1
            if inter_w >0:
                inter_h = min(boxes[n,3],query_boxes[k,3])-max(boxes[n,1],query_boxes[k,1])+1
                if inter_h > 0:
                    # 总面积
                    whole_area = (boxes[n,2]-boxes[n,0]+1)*(boxes[n,3]-boxes[n,1]+1)+box_area-inter_h*inter_w
                    overlaps[n,k] = inter_w*inter_h/whole_area
    return overlaps
