import jittor as jt 
import numpy as np 
import math 

def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]

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
    dw = jt.log(gt_widths / proposals_widths)
    dh = jt.log(gt_heights / proposals_heights)
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
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.)):
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
    dw = jt.log(gw / pw)
    dh = jt.log(gh / ph)
    deltas = jt.stack([dx, dy, dw, dh], dim=-1)

    means = jt.array(means).unsqueeze(0)
    stds = jt.array(stds).unsqueeze(0)
    deltas = (deltas-means)/stds

    return deltas


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000):
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
    means = jt.array(means).repeat(1, deltas.size(1) // 4)
    stds = jt.array(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
    dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
    # Compute center of each roi
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    # Compute width/height of each roi
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
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

    x0,y0,x1,y1,angle=rrect[:5]

    x_ctr = (x0+x1) / 2
    y_ctr = (y0+y1) / 2
    width = x1-x0
    height = y1 - y0
    angle = (angle) / 180 * np.pi

    # x_ctr=x0
    # y_ctr=y0
    # width=x1
    # height=y1
    # angle = (angle) / 180 * np.pi

    # x_ctr, y_ctr, width, height, angle = rrect[:5]

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

def rotated_box_to_bbox_np(rotatex_boxes):
    polys = rotated_box_to_poly_np(rotatex_boxes)
    xmin = polys[:, ::2].min(1, keepdims=True)
    ymin = polys[:, 1::2].min(1, keepdims=True)
    xmax = polys[:, ::2].max(1, keepdims=True)
    ymax = polys[:, 1::2].max(1, keepdims=True)
    return np.concatenate([xmin, ymin, xmax, ymax], axis=1)

def rotated_box_to_bbox(rotatex_boxes):
    polys = rotated_box_to_poly(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return jt.stack([xmin, ymin, xmax, ymax], dim=1)