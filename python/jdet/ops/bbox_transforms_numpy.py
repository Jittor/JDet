import cv2
import numpy as np

pi = np.pi

def poly2obb(polys):
    # [-90 90)
    order = polys.shape[:-1]
    num_points = polys.shape[-1] // 2
    polys = polys.reshape(-1, num_points, 2)
    polys = polys.astype(np.float32)

    obboxes = []
    for poly in polys:
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        if w >= h:
            angle = -angle
        else:
            w, h = h, w
            angle = -90 - angle
        theta = angle / 180 * pi
        obboxes.append([x, y, w, h, theta])

    if not obboxes:
        obboxes = np.zeros((0, 5), dtype=np.float32)
    else:
        obboxes = np.array(obboxes, dtype=np.float32)
    return obboxes.reshape(*order, 5)


def rectpoly2obb(polys):
    theta = np.arctan2(-(polys[..., 3] - polys[..., 1]),
                       polys[..., 2] - polys[..., 0])
    Cos, Sin = np.cos(theta), np.sin(theta)
    Matrix = np.stack([Cos, -Sin, Sin, Cos], axis=-1)
    Matrix = Matrix.reshape(*Matrix.shape[:-1], 2, 2)

    x = polys[..., 0::2].mean(-1)
    y = polys[..., 1::2].mean(-1)
    center = np.stack([x, y], axis=-1).expand_dims(-2)
    center_polys = polys.reshape(*polys.shape[:-1], 4, 2) - center
    rotate_polys = np.matmul(center_polys, Matrix.swapaxes(-1, -2))

    xmin = np.min(rotate_polys[..., :, 0], axis=-1)
    xmax = np.max(rotate_polys[..., :, 0], axis=-1)
    ymin = np.min(rotate_polys[..., :, 1], axis=-1)
    ymax = np.max(rotate_polys[..., :, 1], axis=-1)
    w = xmax - xmin
    h = ymax - ymin

    obboxes = np.stack([x, y, w, h, theta], axis=-1)



def poly2hbb(polys):
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1]//2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


def obb2poly(obboxes):
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate(
        [w/2 * Cos, -w/2 * Sin], axis=-1)
    vector2 = np.concatenate(
        [-h/2 * Sin, -h/2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    return np.concatenate(
        [point1, point2, point3, point4], axis=-1)


def obb2hbb(obboxes):
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    x_bias = np.abs(w/2 * Cos) + np.abs(h/2 * Sin)
    y_bias = np.abs(w/2 * Sin) + np.abs(h/2 * Cos)
    bias = np.concatenate([x_bias, y_bias], axis=-1)
    return np.concatenate([center-bias, center+bias], axis=-1)


def hbb2poly(hbboxes):
    l, t, r, b = [hbboxes[..., i] for i in range(4)]
    return np.stack([l, t, r, t, r, b, l, b], axis=-1)


def hbb2obb(hbboxes):
    order = hbboxes.shape[:-1]
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]

    theta = np.zeros(order, dtype=np.float32)
    obboxes1 = np.stack([x, y, w, h, theta], axis=-1)
    obboxes2 = np.stack([x, y, h, w, theta-pi/2], axis=-1)
    obboxes = np.where((w >= h)[..., None], obboxes1, obboxes2)
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

def get_bbox_type(bboxes, with_score=False):
    dim = bboxes.shape[-1]
    if with_score:
        dim -= 1

    if dim == 4:
        return 'hbb'
    if dim == 5:
        return 'obb'
    if dim == 8:
        return 'poly'
    return 'notype'


def get_bbox_dim(bbox_type, with_score=False):
    if bbox_type == 'hbb':
        dim = 4
    elif bbox_type == 'obb':
        dim = 5
    elif bbox_type == 'poly':
        dim = 8
    else:
        raise ValueError(f"don't know {bbox_type} bbox dim")

    if with_score:
        dim += 1
    return dim


def choice_by_type(hbb_op, obb_op, poly_op, bboxes_or_type,
                   with_score=False):
    if isinstance(bboxes_or_type, np.ndarray):
        bbox_type = get_bbox_type(bboxes_or_type, with_score)
    elif isinstance(bboxes_or_type, str):
        bbox_type = bboxes_or_type
    else:
        raise TypeError(f'need np.ndarray or str,',
                        f'but get {type(bboxes_or_type)}')

    if bbox_type == 'hbb':
        return hbb_op
    elif bbox_type == 'obb':
        return obb_op
    elif bbox_type == 'poly':
        return poly_op
    else:
        raise ValueError('notype bboxes is not suppert')


def regular_theta(theta, mode='180', start=-pi/2):
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def regular_obb(obboxes):
    x, y, w, h, theta = [obboxes[..., i] for i in range(5)]
    w_regular = np.where(w > h, w, h)
    h_regular = np.where(w > h, h, w)
    theta_regular = np.where(w > h, theta, theta+pi/2)
    theta_regular = regular_theta(theta_regular)
    return np.stack([x, y, w_regular, h_regular, theta_regular], axis=-1)
