import jittor as jt
import numpy as np      #TODO: remove numpy
import cv2
import math

def dbbox2delta_v3(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    proposals = proposals.float()
    gt = gt.float()
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (jt.cos(proposals[..., 4]) * coord[..., 0] +
          jt.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
    dy = (-jt.sin(proposals[..., 4]) * coord[..., 0] +
          jt.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
    dw = jt.log(gt_widths / proposals_widths)
    dh = jt.log(gt_heights / proposals_heights)
    dangle = gt_angle - proposals_angle
    deltas = jt.stack((dx, dy, dw, dh, dangle), -1)

    means = jt.array(means).unsqueeze(0)
    stds = jt.array(stds).unsqueeze(0)
    deltas = (deltas - means) / stds

    return deltas

def hbb2obb_v2(boxes):
    num_boxes = boxes.size(0)
    ex_heights = boxes[..., 2] - boxes[..., 0] + 1.0
    ex_widths = boxes[..., 3] - boxes[..., 1] + 1.0
    ex_ctr_x = boxes[..., 0] + 0.5 * (ex_heights - 1.0)
    ex_ctr_y = boxes[..., 1] + 0.5 * (ex_widths - 1.0)
    c_bboxes = jt.contrib.concat((ex_ctr_x.unsqueeze(1), ex_ctr_y.unsqueeze(1), ex_widths.unsqueeze(1), ex_heights.unsqueeze(1)), 1)
    initial_angles = -jt.ones((num_boxes, 1)) * np.pi / 2
    dbboxes = jt.contrib.concat((c_bboxes, initial_angles), 1)

    return dbboxes

def mask2poly_single(binary_mask):
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    poly = cv2.boxPoints(rect)
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    return list(polys)

def polygonToRotRectangle_batch(bbox, with_module=True):
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
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
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def get_best_begin_point_warp_single(coordinate):
    return TuplePoly2Poly(get_best_begin_point_single(coordinate))

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_warp_single, coordinate_list)
    best_coordinate_list = np.stack(list(best_coordinate_list))
    return best_coordinate_list

def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
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

    means = jt.float32(means).unsqueeze(0)
    stds = jt.float32(stds).unsqueeze(0)
    deltas = deltas.subtract(means).divide(stds)

    return deltas

def dbbox2delta_v2(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    roi_widths = proposals[..., 2]
    roi_heights = proposals[..., 3]
    roi_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    targets_dx = (jt.cos(roi_angle) * coord[..., 0] + jt.sin(roi_angle) * coord[:, 1]) / roi_widths
    targets_dy = (-jt.sin(roi_angle) * coord[..., 0] + jt.cos(roi_angle) * coord[:, 1]) / roi_heights
    targets_dw = jt.log(gt_widths / roi_widths)
    targets_dh = jt.log(gt_heights / roi_heights)
    targets_dangle = (gt_angle - roi_angle)
    dist = targets_dangle % (2 * np.pi)
    dist = jt.min(dist, np.pi * 2 - dist)
    try:
        assert np.all(dist.numpy() <= (np.pi/2. + 0.001) )
    except:
        import pdb
        pdb.set_trace()

    inds = jt.sin(targets_dangle) < 0
    dist[inds] = -dist[inds]
    # TODO: change the norm value
    dist = dist / (np.pi / 2.)
    deltas = jt.stack((targets_dx, targets_dy, targets_dw, targets_dh, dist), -1)


    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = (deltas - means) / stds

    return deltas

def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
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
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = px + pw * dx  # gx = px + pw * dx
    gy = py + ph * dy  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min_v=0, max_v=max_shape[1] - 1)
        y1 = y1.clamp(min_v=0, max_v=max_shape[0] - 1)
        x2 = x2.clamp(min_v=0, max_v=max_shape[1] - 1)
        y2 = y2.clamp(min_v=0, max_v=max_shape[0] - 1)
    bboxes = jt.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes

def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = jt.ones((bboxes.size(0), 1), dtype=bboxes.dtype) * img_id
            rois = jt.contrib.concat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = jt.contrib.concat(rois_list, 0)
    return rois