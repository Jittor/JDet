import cv2
import numpy as np
import jittor as jt

pi = 3.141592

def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def regular_obb(obboxes):
    x, y, w, h, theta = obboxes.unbind(dim=-1)
    w_regular = jt.ternary(w > h, w, h)
    h_regular = jt.ternary(w > h, h, w)
    theta_regular = jt.ternary(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return jt.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)

def get_bbox_type(bboxes, with_score=False):

    dim = bboxes.size(-1)

    if with_score:
        dim -= 1
    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim  == 8:
        return 'poly'
    return 'notype'

def poly2obb(polys):

    polys_np = polys.numpy()

    order = polys_np.shape[:-1]
    num_points = polys_np.shape[-1] // 2
    polys_np = polys_np.reshape(-1, num_points, 2)
    polys_np = polys_np.astype(np.float32)

    obboxes = []
    for poly in polys_np:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * pi
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5))
    else:
        obboxes = np.array(obboxes)

    obboxes = obboxes.reshape(*order, 5)
    return polys.new_tensor(obboxes)


def rectpoly2obb(polys):
    theta = jt.atan2(-(polys[..., 3] - polys[..., 1]),
                        polys[..., 2] - polys[..., 0])
    Cos, Sin = jt.cos(theta), jt.sin(theta)
    Matrix = jt.stack([Cos, -Sin, Sin, Cos], dim=-1)
    Matrix = Matrix.view(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = jt.stack([x, y], dim=-1).unsqueeze(-2)
    center_polys = polys.view(*polys.shape[:-1], 4, 2) - center
    rotate_polys = jt.matmul(center_polys, Matrix.transpose(-1, -2))

    xmin = jt.min(rotate_polys[..., :, 0], dim=-1)
    xmax = jt.max(rotate_polys[..., :, 0], dim=-1)
    ymin = jt.min(rotate_polys[..., :, 1], dim=-1)
    ymax = jt.max(rotate_polys[..., :, 1], dim=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = jt.stack([x, y, w, h, theta], dim=-1)
    return regular_obb(obboxes)


def poly2hbb(polys):

    polys = polys.view(*polys.shape[:-1], polys.size(-1)//2, 2)

    lt_point = jt.min(polys, dim=-2)
    rb_point = jt.max(polys, dim=-2)

    return jt.concat([lt_point, rb_point], dim=-1)


def obb2poly(obboxes):
    center, w, h, theta = jt.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = jt.cos(theta), jt.sin(theta)

    vector1 = jt.concat([w/2 * Cos, -w/2 * Sin], dim=-1)
    vector2 = jt.concat([-h/2 * Sin, -h/2 * Cos], dim=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return jt.concat([point1, point2, point3, point4], dim=-1)


def obb2hbb(obboxes):
    center, w, h, theta = jt.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = jt.cos(theta), jt.sin(theta)
    x_bias = jt.abs(w/2 * Cos) + jt.abs(h/2 * Sin)
    y_bias = jt.abs(w/2 * Sin) + jt.abs(h/2 * Cos)
    bias = jt.concat([x_bias, y_bias], dim=-1)
    return jt.concat([center-bias, center+bias], dim=-1)


def hbb2poly(hbboxes):
    l, t, r, b = hbboxes.unbind(-1)
    return jt.stack([l, t, r, t, r, b, l ,b], dim=-1)


def hbb2obb(hbboxes):
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)

    obboxes1 = jt.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = jt.stack([x, y, h, w, theta-pi/2], dim=-1)
    obboxes = jt.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes

_type_func_map = {
    ('poly', 'obb'): poly2obb,
    ('poly', 'hbb'): poly2hbb,
    ('obb', 'poly'): obb2poly,
    ('obb', 'hbb'): obb2hbb,
    ('hbb', 'poly'): hbb2poly,
    ('hbb', 'obb'): hbb2obb
}

def bbox2type(bboxes, to_type):
    assert to_type in ['hbb', 'obb', 'poly']

    ori_type = get_bbox_type(bboxes)
    if ori_type == 'notype':
        raise ValueError('Not a bbox type')
    if ori_type == to_type:
        return bboxes
    trans_func = _type_func_map[(ori_type, to_type)]
    return trans_func(bboxes)

def get_bbox_areas(bboxes):
    btype = get_bbox_type(bboxes)
    if btype == 'hbb':
        wh = bboxes[..., 2:] - bboxes[..., :2]
        areas = wh[..., 0] * wh[..., 1]
    elif btype == 'obb':
        areas = bboxes[..., 2] * bboxes[..., 3]
    elif btype == 'poly':
        pts = bboxes.view(*bboxes.size()[:-1], 4, 2)
        roll_pts = torch.roll(pts, 1, dims=-2)
        xyxy = torch.sum(pts[..., 0] * roll_pts[..., 1] -
                         roll_pts[..., 0] * pts[..., 1], dim=-1)
        areas = 0.5 * torch.abs(xyxy)
    else:
        raise ValueError('The type of bboxes is notype')

    return areas