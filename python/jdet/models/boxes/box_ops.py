import jittor as jt 
import numpy as np 
import math 

def loc2bbox(src_bbox,loc,mean=[0.,0.,0.,0.],std=[1.,1.,1.,1.]):
    if src_bbox.shape[0] == 0:
        return jt.zeros((0, 4), dtype=loc.dtype)
    

    mean = jt.array(mean)
    std = jt.array(std)
    loc = loc*std+mean 

    src_width = src_bbox[:, 2:3] - src_bbox[:, 0:1]
    src_height = src_bbox[:, 3:4] - src_bbox[:, 1:2]
    src_center_x = src_bbox[:, 0:1] + 0.5 * src_width
    src_center_y = src_bbox[:, 1:2] + 0.5 * src_height

    dx = loc[:, 0:1]
    dy = loc[:, 1:2]
    dw = loc[:, 2:3]
    dh = loc[:, 3:4]

    center_x = dx*src_width+src_center_x
    center_y = dy*src_height+src_center_y
        
    w = jt.exp(dw) * src_width
    h = jt.exp(dh) * src_height
        
    x1,y1,x2,y2 = center_x-0.5*w, center_y-0.5*h, center_x+0.5*w, center_y+0.5*h
        
    dst_bbox = jt.contrib.concat([x1,y1,x2,y2],dim=1)

    return dst_bbox

def loc2bbox_r(src_bbox,loc,mean=[0.,0.,0.,0.,0.],std=[1.,1.,1.,1.,1.]):
    if src_bbox.shape[0] == 0:
        return jt.zeros((0, 4), dtype=loc.dtype)
    

    mean = jt.array(mean)
    std = jt.array(std)
    loc = loc*std+mean 

    src_center_x = src_bbox[:, 0:1]
    src_center_y = src_bbox[:, 1:2]
    src_width = src_bbox[:, 2:3]
    src_height = src_bbox[:, 3:4]

    dx = loc[:, 0:1]
    dy = loc[:, 1:2]
    dw = loc[:, 2:3]
    dh = loc[:, 3:4]

    center_x = dx*src_width+src_center_x
    center_y = dy*src_height+src_center_y
        
    w = jt.exp(dw) * src_width
    h = jt.exp(dh) * src_height
        
    x1,y1,x2,y2 = center_x-0.5*w, center_y-0.5*h, center_x+0.5*w, center_y+0.5*h
    theta = loc[:, 4:5] + src_bbox[:, 4:5]
    dst_bbox = jt.contrib.concat([(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1,theta],dim=1)
    return dst_bbox
    
def bbox2loc_r(src_bbox,dst_bbox,mean=[0.,0.,0.,0.,0.],std=[1.,1.,1.,1.,1.]):        
    center_x, center_y, width, height = src_bbox[:, 0:1], src_bbox[:, 1:2], src_bbox[:, 2:3], src_bbox[:, 3:4]
    base_center_x, base_center_y, base_width, base_height = dst_bbox[:, 0:1], dst_bbox[:, 1:2], dst_bbox[:, 2:3], dst_bbox[:, 3:4]

    eps = 1e-5

    dx = (base_center_x - center_x) / (width + 1)
    dy = (base_center_y - center_y) / (height + 1)

    dw = jt.log(base_width / (width + 1) + eps)
    dh = jt.log(base_height / (height + 1) + eps)

    da = dst_bbox[:, 4:5] - src_bbox[:, 4:5]
        
    loc = jt.contrib.concat([dx,dy,dw,dh,da],dim=1)

    mean = jt.array(mean)
    std = jt.array(std)
    loc = (loc-mean)/std 

    return loc
    
def bbox2loc(src_bbox,dst_bbox,mean=[0.,0.,0.,0.],std=[1.,1.,1.,1.]):        
    width = src_bbox[:, 2:3] - src_bbox[:, 0:1]
    height = src_bbox[:, 3:4] - src_bbox[:, 1:2]
    center_x = src_bbox[:, 0:1] + 0.5 * width
    center_y = src_bbox[:, 1:2] + 0.5 * height

    base_width = dst_bbox[:, 2:3] - dst_bbox[:, 0:1]
    base_height = dst_bbox[:, 3:4] - dst_bbox[:, 1:2]
    base_center_x = dst_bbox[:, 0:1] + 0.5 * base_width
    base_center_y = dst_bbox[:, 1:2] + 0.5 * base_height

    eps = 1e-5
    height = jt.maximum(height, eps)
    width = jt.maximum(width, eps)

    dy = (base_center_y - center_y) / height
    dx = (base_center_x - center_x) / width

    dw = jt.safe_log(base_width / width)
    dh = jt.safe_log(base_height / height)
        
    loc = jt.contrib.concat([dx,dy,dw,dh],dim=1)

    mean = jt.array(mean)
    std = jt.array(std)
    loc = (loc-mean)/std 

    return loc
    
def bbox_iou(bbox_a, bbox_b):
    assert bbox_a.shape[1]==4 and bbox_b.shape[1]==4
    if bbox_a.numel()==0 or bbox_b.numel()==0:
        return jt.zeros((bbox_a.shape[0],bbox_b.shape[0]))
    # top left
    tl = jt.maximum(bbox_a[:, :2].unsqueeze(1), bbox_b[:, :2])
    # bottom right
    br = jt.minimum(bbox_a[:,2:].unsqueeze(1), bbox_b[:, 2:])

    area_i = jt.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = jt.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = jt.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    return area_i / (area_a.unsqueeze(1) + area_b - area_i)


def bbox_iou_per_box(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.transpose(1,0)

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (jt.minimum(b1_x2, b2_x2) - jt.maximum(b1_x1, b2_x1)).clamp(0) * \
            (jt.minimum(b1_y2, b2_y2) - jt.maximum(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = jt.maximum(b1_x2, b2_x2) - jt.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = jt.maximum(b1_y2, b2_y2) - jt.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * jt.pow(jt.atan(w2 / h2) - jt.atan(w1 / h1), 2)
                with jt.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def norm_angle(angle, range=[float(-np.pi / 4), float(np.pi)]):
    ret = (angle - range[0]) % range[1] + range[0]
    return ret

def bbox2delta_rotated(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 5)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 5)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 5), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angles = proposals[..., 4]

    cosa = jt.cos(proposals_angles)
    sina = jt.sin(proposals_angles)
    coord = gt[..., 0:2] - proposals[..., 0:2]

    dx = (cosa * coord[..., 0] + sina * coord[..., 1]) / proposals_widths
    dy = (-sina * coord[..., 0] + cosa * coord[..., 1]) / proposals_heights
    dw = jt.safe_log(gt_widths / proposals_widths)
    dh = jt.safe_log(gt_heights / proposals_heights)
    da = (gt_angle - proposals_angles)
    da = norm_angle(da) / np.pi

    deltas = jt.stack((dx, dy, dw, dh, da), -1)

    means = jt.array(means)
    means = jt.array(means).unsqueeze(0)
    stds = jt.array(stds).unsqueeze(0)
    deltas = (deltas-means)/stds

    return deltas


def delta2bbox_rotated(rois, deltas, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), max_shape=None,
                       wh_ratio_clip=16 / 1000, clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 5)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 5 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 5), where columns represent

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """
    means = jt.array(means).repeat(1, deltas.size(1) // 5)
    stds = jt.array(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
    dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
    roi_x = (rois[:, 0]).unsqueeze(1).expand_as(dx)
    roi_y = (rois[:, 1]).unsqueeze(1).expand_as(dy)
    roi_w = (rois[:, 2]).unsqueeze(1).expand_as(dw)
    roi_h = (rois[:, 3]).unsqueeze(1).expand_as(dh)
    roi_angle = (rois[:, 4]).unsqueeze(1).expand_as(dangle)
    gx = dx * roi_w * jt.cos(roi_angle) \
         - dy * roi_h * jt.sin(roi_angle) + roi_x
    gy = dx * roi_w * jt.sin(roi_angle) \
         + dy * roi_h * jt.cos(roi_angle) + roi_y
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    bboxes = jt.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes


def bbox2delta(proposals,
               gt,
               means=None,
               stds=None,
               weights = None):
    """Compute deltas of proposals w.r.t. gt in the MMDet V1.x manner.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of `delta2bbox()`

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = jt.safe_log(gw / pw)
    dh = jt.safe_log(gh / ph)
    deltas = jt.stack([dx, dy, dw, dh], dim=-1)

    if means is not None and stds is not None:
        means = jt.array(means).unsqueeze(0)
        stds = jt.array(stds).unsqueeze(0)
        deltas = (deltas-means)/stds
    
    if weights is not None:
        assert deltas.shape[-1] == weights.shape[-1]
        weights = jt.array(weights)
        deltas*=weights

    return deltas

def delta2bbox(rois,
               deltas,
               means=None,
               stds=None,
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               weights = None,):
    """Apply deltas to shift/scale base boxes in the MMDet V1.x manner.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of `bbox2delta()`

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 4 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.

    Returns:
        Tensor: Boxes with shape (N, 4), where columns represent
            tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = jt.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = jt.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> legacy_delta2bbox(rois, deltas, max_shape=(32, 32))
        tensor([[0.0000, 0.0000, 1.5000, 1.5000],
                [0.0000, 0.0000, 5.2183, 5.2183],
                [0.0000, 0.1321, 7.8891, 0.8679],
                [5.3967, 2.4251, 6.0033, 3.7749]])
    """
    if weights is not None:
        assert deltas.shape[-1] == len(weights)
        weights = jt.array(weights)
        deltas /= weights

    if means is not None and stds is not None:
        means = jt.array(means).view(1, -1).repeat(1, deltas.size(-1) // 4)
        stds = jt.array(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
        deltas = deltas * stds + means
    dx = deltas[..., 0::4]
    dy = deltas[..., 1::4]
    dw = deltas[..., 2::4]
    dh = deltas[..., 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
    dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
    # Compute center of each roi
    px = ((rois[..., 0] + rois[..., 2]) *
          0.5).unsqueeze(-1)  # .expand_as(dx)
    py = ((rois[..., 1] + rois[..., 3]) *
          0.5).unsqueeze(-1)  # .expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[..., 2] - rois[..., 0]).unsqueeze(-1)  # .expand_as(dw)
    ph = (rois[..., 3] - rois[..., 1]).unsqueeze(-1)  # .expand_as(dh)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right

    # The true legacy box coder should +- 0.5 here.
    # However, current implementation improves the performance when testing
    # the models trained in MMDetection 1.X (~0.5 bbox AP, 0.2 mask AP)
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    if max_shape is not None:
        x1 = x1.clamp(min_v=0, max_v=max_shape[1] - 1)
        y1 = y1.clamp(min_v=0, max_v=max_shape[0] - 1)
        x2 = x2.clamp(min_v=0, max_v=max_shape[1] - 1)
        y2 = y2.clamp(min_v=0, max_v=max_shape[0] - 1)
    bboxes = jt.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def poly_to_rotated_box_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_box:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    width = max(edge1, edge2)
    height = min(edge1, edge2)

    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    angle = norm_angle(angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rotated_box = np.array([x_ctr, y_ctr, width, height, angle])
    return rotated_box

def poly_to_rotated_box_np(polys):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rotated_boxes:[x_ctr,y_ctr,w,h,angle]
    """
    rotated_boxes = []
    for poly in polys:
        rotated_box = poly_to_rotated_box_single(poly)
        rotated_boxes.append(rotated_box)
    return np.array(rotated_boxes).astype(np.float32)


def poly_to_rotated_box(polys):
    """
    polys:n*8
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)

    edge1 = jt.sqrt(
        jt.pow(pt1[..., 0] - pt2[..., 0], 2) + jt.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = jt.sqrt(
        jt.pow(pt2[..., 0] - pt3[..., 0], 2) + jt.pow(pt2[..., 1] - pt3[..., 1], 2))

    angles1 = jt.atan2((pt2[..., 1] - pt1[..., 1]), (pt2[..., 0] - pt1[..., 0]))
    angles2 = jt.atan2((pt4[..., 1] - pt1[..., 1]), (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]

    angles = norm_angle(angles)

    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0

    edges = jt.stack([edge1, edge2], dim=1)
    width = jt.max(edges, 1)
    height = jt.min(edges, 1)

    return jt.stack([x_ctr, y_ctr, width, height, angles], 1)

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)


def get_best_begin_point(coordinates):
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def rotated_box_to_poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly

def rotated_box_to_poly_np(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if rrects.shape[0] == 0:
        return np.zeros([0,8], dtype=np.float32)
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle = rrect[:5]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys.astype(np.float32)
    
def rotated_box_to_poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    n = rrects.shape[0]
    if n == 0:
        return jt.zeros([0,8])
    x_ctr = rrects[:, 0] 
    y_ctr = rrects[:, 1] 
    width = rrects[:, 2] 
    height = rrects[:, 3] 
    angle = rrects[:, 4]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = jt.stack([tl_x, br_x, br_x, tl_x, tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y, tl_y, tl_y, br_y, br_y], 1).reshape([n, 2, 8])
    c = jt.cos(angle)
    s = jt.sin(angle)
    R = jt.stack([c, c, c, c, s, s, s, s, -s, -s, -s, -s, c, c, c, c], 1).reshape([n, 2, 8])
    offset = jt.stack([x_ctr, x_ctr, x_ctr, x_ctr, y_ctr, y_ctr, y_ctr, y_ctr], 1)
    poly = ((R * rect).sum(1) + offset).reshape([n, 2, 4]).permute([0,2,1]).reshape([n, 8])
    return poly


def rotated_box_to_bbox_np(rotatex_boxes):
    if rotatex_boxes.shape[0]==0:
        return np.zeros((0,4)),np.zeros((0,8))
    polys = rotated_box_to_poly_np(rotatex_boxes)
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1),polys

# def rotated_box_to_poly(rboxes):
#     """
#     rrect:[x_ctr,y_ctr,w,h,angle]
#     to
#     poly:[x0,y0,x1,y1,x2,y2,x3,y3]
#     """
#     N = rboxes.shape[0]
#     x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
#         1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
#     tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5

#     rects = jt.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y,
#                          br_y, br_y], dim=0).reshape(2, 4, N).permute(2, 0, 1)

#     sin, cos = jt.sin(angle), jt.cos(angle)
#     # M.shape=[N,2,2]
#     M = jt.stack([cos, -sin, sin, cos],dim=0).reshape(2, 2, N).permute(2, 0, 1)
#     # polys:[N,8]
#     polys = jt.matmul(M,rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
#     polys[:, ::2] += x_ctr.unsqueeze(1)
#     polys[:, 1::2] += y_ctr.unsqueeze(1)

#     return polys

def rotated_box_to_bbox(rotatex_boxes):
    polys = rotated_box_to_poly(rotatex_boxes)
    xmin = polys[:, ::2].min(1)
    ymin = polys[:, 1::2].min(1)
    xmax = polys[:, ::2].max(1)
    ymax = polys[:, 1::2].max(1)
    return jt.stack([xmin, ymin, xmax, ymax], dim=1)
    # return jt.stack([(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin], dim=1)

def boxes_xywh_to_x0y0x1y1(boxes):
    assert(boxes.shape[1] >= 4)
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    others = boxes[:, 4:]
    return jt.concat([jt.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], dim=1), others], dim=1)

def boxes_x0y0x1y1_to_xywh(boxes):
    assert(boxes.shape[1] >= 4)
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    others = boxes[:, 4:]
    return jt.concat([jt.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=1), others], dim=1)

from jdet.ops.bbox_transforms import regular_obb, regular_theta

def mintheta_obb(obboxes):
    pi = 3.141592
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    theta1 = regular_theta(theta)
    theta2 = regular_theta(theta + pi/2)
    abs_theta1 = jt.abs(theta1)
    abs_theta2 = jt.abs(theta2)

    w_regular = jt.ternary(abs_theta1 < abs_theta2, w, h)
    h_regular = jt.ternary(abs_theta1 < abs_theta2, h, w)
    theta_regular = jt.ternary(abs_theta1 < abs_theta2, theta1, theta2)

    obboxes = jt.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)
    return obboxes

def distance2obb(points, distance, max_shape=None):
    distance, theta = distance.split([4, 1], dim=1)

    Cos, Sin = jt.cos(theta), jt.sin(theta)
    Matrix = jt.concat([Cos, Sin, -Sin, Cos], dim=1).reshape(-1, 2, 2)

    wh = distance[:, :2] + distance[:, 2:]
    offset_t = (distance[:, 2:] - distance[:, :2]) / 2
    offset_t = offset_t.unsqueeze(2)
    offset = jt.nn.bmm(Matrix, offset_t).squeeze(2)
    ctr = points + offset

    obbs = jt.concat([ctr, wh, theta], dim=1)
    return regular_obb(obbs)