import jittor as jt
import numpy as np      #TODO: remove numpy
import cv2
import math
import copy             #TODO: remove copy(?)

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
    res = cv2.findContours(binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #TODO: check binary_mask.copy()
    if (len(res) == 2):
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    poly = cv2.boxPoints(rect)
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    return list(polys)

def obb2poly_single(gt_obb):
    '''
    Args:
        gt_obb (np.array): [x, y, h, w, a]
    Rets:
        poly (np.array): shape is [4, 2]
    '''
    center = np.array([gt_obb[0], gt_obb[1]])
    vx = np.array([(gt_obb[2] - 1)*math.cos(gt_obb[4])/2, (gt_obb[2] - 1)*math.sin(gt_obb[4])/2])
    vy = np.array([(gt_obb[3] - 1)*math.sin(gt_obb[4])/2, -(gt_obb[3] - 1)*math.cos(gt_obb[4])/2])
    p1 = center + vx + vy
    p2 = center + vx - vy
    p3 = center - vx - vy
    p4 = center - vx + vy
    return np.stack((p1, p2, p3, p4))

def obb2poly(gt_obb_list):
    polys = map(obb2poly_single, gt_obb_list)
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
    dist = jt.minimum(dist, np.pi * 2 - dist)
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


    means = jt.array(means).unsqueeze(0)
    stds = jt.array(stds).unsqueeze(0)
    deltas = (deltas - means) / stds

    return deltas

def choose_best_match_batch(Rrois, gt_rois):
    Rroi_angles = Rrois[:, 4].unsqueeze(1)

    gt_xs, gt_ys, gt_ws, gt_hs, gt_angles = gt_rois[:, 0].copy(), gt_rois[:, 1].copy(), gt_rois[:, 2].copy(), gt_rois[:, 3].copy(), gt_rois[:, 4].copy()

    gt_angle_extent = jt.contrib.concat((gt_angles[:, np.newaxis], (gt_angles + np.pi/2.)[:, np.newaxis],
                                      (gt_angles + np.pi)[:, np.newaxis], (gt_angles + np.pi * 3/2.)[:, np.newaxis]), 1)
    dist = (Rroi_angles - gt_angle_extent) % (2 * np.pi)
    dist = jt.minimum(dist, np.pi * 2 - dist)
    min_index = jt.argmin(dist, 1)[0]

    gt_rois_extent0 = copy.deepcopy(gt_rois)
    gt_rois_extent1 = jt.contrib.concat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi/2.), 1)
    gt_rois_extent2 = jt.contrib.concat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_ws.unsqueeze(1), gt_hs.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi), 1)
    gt_rois_extent3 = jt.contrib.concat((gt_xs.unsqueeze(1), gt_ys.unsqueeze(1), \
                                 gt_hs.unsqueeze(1), gt_ws.unsqueeze(1), gt_angles.unsqueeze(1) + np.pi * 3/2.), 1)
    gt_rois_extent = jt.contrib.concat((gt_rois_extent0.unsqueeze(1),
                                     gt_rois_extent1.unsqueeze(1),
                                     gt_rois_extent2.unsqueeze(1),
                                     gt_rois_extent3.unsqueeze(1)), 1)

    gt_rois_new = jt.zeros_like(gt_rois)
    for curiter, index in enumerate(min_index):
        gt_rois_new[curiter, :] = gt_rois_extent[curiter, index.item(), :]

    gt_rois_new[:, 4] = gt_rois_new[:, 4] % (2 * np.pi)

    return gt_rois_new

def best_match_dbbox2delta(Rrois, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    gt_boxes_new = choose_best_match_batch(Rrois, gt)
    try:
        assert np.all(Rrois.numpy()[:, 4] <= (np.pi + 0.001))
    except:
        import pdb
        pdb.set_trace()
    bbox_targets = dbbox2delta_v2(Rrois, gt_boxes_new, means, stds)

    return bbox_targets

def dbbox2result(dbboxes, labels, num_classes):
    # labels + 1 for dotadataset evaluation and data_merge.
    dbboxes = dbboxes.numpy()
    labels = labels.numpy()
    return dbboxes[:, :8], dbboxes[:, -1].flatten(), labels + 1

def delta2dbbox_v3(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    means = jt.array(means, dtype=deltas.dtype).repeat(1, deltas.size(1) // 5)
    stds = jt.array(stds, dtype=deltas.dtype).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
    dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    # import pdb
    # pdb.set_trace()
    gx = dx * Rroi_w * jt.cos(Rroi_angle) \
         - dy * Rroi_h * jt.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * jt.sin(Rroi_angle) \
         + dy * Rroi_h * jt.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    # TODO: check the hard code
    # gangle = (2 * np.pi) * dangle + Rroi_angle
    gangle = dangle + Rroi_angle
    # gangle = gangle % ( 2 * np.pi)

    if max_shape is not None:
        pass

    bboxes = jt.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes

def delta2dbbox_v2(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    means = jt.array(means, dtype=deltas.dtype).repeat(1, deltas.size(1) // 5)
    stds = jt.array(stds, dtype=deltas.dtype).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
    dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    gx = dx * Rroi_w * jt.cos(Rroi_angle) \
         - dy * Rroi_h * jt.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * jt.sin(Rroi_angle) \
         + dy * Rroi_h * jt.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    gangle = (np.pi / 2.) * dangle + Rroi_angle

    if max_shape is not None:
        pass

    bboxes = jt.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes

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

def gt_mask_bp_obbs(gt_masks, with_module=True):

    # trans gt_masks to gt_obbs
    gt_polys = mask2poly(gt_masks)
    gt_bp_polys = get_best_begin_point(gt_polys)
    gt_obbs = polygonToRotRectangle_batch(gt_bp_polys, with_module)

    return gt_obbs

def gt_mask_bp_obbs_list(gt_masks_list):

    gt_obbs_list = map(gt_mask_bp_obbs, gt_masks_list)

    return list(gt_obbs_list)

def roi2droi(rois):
    """
    :param rois: Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    :return: drois: Tensor: shape (n, 6), [batch_ind, x, y, w, h, theta]
    """
    hbbs = rois[:, 1:]
    obbs = hbb2obb_v2(hbbs)

    return jt.contrib.concat((rois[:, 0].unsqueeze(1), obbs), 1)

def choose_best_Rroi_batch(Rroi):
    """
    There are many instances with large aspect ratio, so we choose the point, previous is long side,
    after is short side, so it makes sure h < w
    then angle % 180,
    :param Rroi: (x_ctr, y_ctr, w, h, angle)
            shape: (n, 5)
    :return: Rroi_new: Rroi with new representation
    """
    x_ctr, y_ctr, w, h, angle = copy.deepcopy(Rroi[:, 0]), copy.deepcopy(Rroi[:, 1]), \
                                copy.deepcopy(Rroi[:, 2]), copy.deepcopy(Rroi[:, 3]), copy.deepcopy(Rroi[:, 4])
    indexes = w < h

    Rroi[indexes, 2] = h[indexes]
    Rroi[indexes, 3] = w[indexes]
    Rroi[indexes, 4] = Rroi[indexes, 4] + np.pi / 2.
    # TODO: check the module
    Rroi[:, 4] = Rroi[:, 4] % np.pi

    return Rroi

def dbbox2roi(dbbox_list):
    """
    Convert a list of dbboxes to droi format.
    :param dbbox_list: (list[Tensor]): a list of dbboxes corresponding to a batch of images
    :return: Tensor: shape (n, 6) [batch_ind, x_ctr, y_ctr, w, h, angle]
    """
    drois_list = []
    for img_id, dbboxes in enumerate(dbbox_list):
        if dbboxes.size(0) > 0:
            img_inds = jt.full((dbboxes.size(0), 1), img_id, dtype=dbboxes.dtype)
            drois = jt.contrib.concat([img_inds, dbboxes[:, :5]], dim=-1)
        else:
            drois = jt.zeros((0, 6), dtype=dbboxes.dtype)

        drois_list.append(drois)
    drois = jt.contrib.concat(drois_list, 0)
    return drois