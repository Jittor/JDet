import jittor as jt 

def nms_v0(dets,thresh):
    '''
      dets jt.array [x1,y1,x2,y2,score]
      x(:,0)->x1,x(:,1)->y1,x(:,2)->x2,x(:,3)->y2,x(:,4)->score
    '''
    threshold = str(thresh)
    order = jt.argsort(dets[:,4],descending=True)[0]
    dets = dets[order]
    s_1 = '(@x(j,2)-@x(j,0))*(@x(j,3)-@x(j,1))'
    s_2 = '(@x(i,2)-@x(i,0))*(@x(i,3)-@x(i,1))'
    s_inter_w = 'max((Tx)0,min(@x(j,2),@x(i,2))-max(@x(j,0),@x(i,0)))'
    s_inter_h = 'max((Tx)0,min(@x(j,3),@x(i,3))-max(@x(j,1),@x(i,1)))'
    s_inter = s_inter_h+'*'+s_inter_w
    iou = s_inter + '/(' + s_1 +'+' + s_2 + '-' + s_inter + ')'
    fail_cond = iou+'>'+threshold
    selected = jt.candidate(dets, fail_cond)
    return order[selected]

def nms(boxes,scores,thresh):
    assert boxes.shape[-1]==4 and len(scores)==len(boxes)
    if scores.ndim==1:
        scores = scores.unsqueeze(-1)
    dets = jt.concat([boxes,scores],dim=1)
    return jt.nms(dets,thresh)

def multiclass_nms(mlvl_bboxes, mlvl_scores, score_thr, nms, max_per_img):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Var): shape (n, #class*4) or (n, 4)
        multi_scores (Var): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.

    Returns:
        tuple: (dets, labels), Var of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    boxes = []
    scores = []
    labels = []
    n_class = mlvl_scores.size(1)
    if mlvl_bboxes.shape[1] > 4:
        mlvl_bboxes = mlvl_bboxes.view(mlvl_bboxes.size(0), -1, 4)
    else:
        mlvl_bboxes = mlvl_bboxes.unsqueeze(1)
        mlvl_bboxes = mlvl_bboxes.expand((mlvl_bboxes.size(0), n_class, 4))
    for j in range(1, n_class):
        bbox_j = mlvl_bboxes[:, j, :]
        score_j = mlvl_scores[:, j:j+1]
        mask = jt.where(score_j > score_thr)[0]
        bbox_j = bbox_j[mask, :]
        score_j = score_j[mask]
        dets = jt.concat([bbox_j, score_j], dim=1)
        keep = jt.nms(dets, nms['iou_threshold'])
        bbox_j = bbox_j[keep]
        score_j = score_j[keep]
        label_j = jt.ones_like(score_j).int32()*j
        boxes.append(bbox_j)
        scores.append(score_j)
        labels.append(label_j)

    boxes = jt.concat(boxes, dim=0)
    scores = jt.concat(scores, dim=0)
    index, _ = jt.argsort(scores, dim=0, descending=True)
    index = index[:max_per_img, 0]
    boxes = jt.concat([boxes, scores], dim=1)[index]
    labels = jt.concat(labels, dim=0).squeeze(1)[index]
    return boxes, labels
