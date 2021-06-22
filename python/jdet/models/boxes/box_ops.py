

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
