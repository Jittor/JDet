from jdet.ops.bbox_transforms import *
from .box_ops import delta2bbox,bbox2delta,delta2bbox_rotated,bbox2delta_rotated
from jdet.utils.registry import BOXES

import jittor as jt
import math

@BOXES.register_module()
class DeltaXYWHBBoxCoder:
    """Delta XYWH BBox coder used in MMDet V1.x.

    Following the practice in R-CNN [1]_, this coder encodes bbox (x1, y1, x2,
    y2) into delta (dx, dy, dw, dh) and decodes delta (dx, dy, dw, dh)
    back to original bbox (x1, y1, x2, y2).

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Args:
        target_means (Sequence[float]): denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 weights=None):
        self.means = target_means
        self.stds = target_stds
        self.weights = None

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (jt.Var): source boxes, e.g., object pred_bboxes.
            gt_bboxes (jt.Var): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            jt.Var: Box transformation deltas
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means,
                                    self.stds,weights=self.weights)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (jt.Var): Basic boxes.
            pred_bboxes (jt.Var): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            jt.Var: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means,
                                    self.stds, max_shape, wh_ratio_clip,weights=self.weights)

        return decoded_bboxes

@BOXES.register_module()
class DeltaXYWHABBoxCoder:
    """Delta XYWHA BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x,y,w,h,a) into delta (dx, dy, dw, dh,da) and
    decodes delta (dx, dy, dw, dh,da) back to original bbox (x, y, w, h, a).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.),
                 clip_border=True):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (jt.Var): Source boxes, e.g., object pred_bboxes.
            gt_bboxes (jt.Var): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            jt.Var: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5
        encoded_bboxes = bbox2delta_rotated(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (jt.Var): Basic boxes.
            pred_bboxes (jt.Var): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            jt.Var: Decoded boxes.
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        decoded_bboxes = delta2bbox_rotated(bboxes, pred_bboxes, self.means, self.stds,
                                            max_shape, wh_ratio_clip, self.clip_border)

        return decoded_bboxes

@BOXES.register_module()
class GVFixCoder:
    def __init__(self):
        pass

    def encode(self, polys):

        assert polys.size(1) == 8
        
        max_x_idx, max_x = polys[:,  ::2].argmax(1)
        min_x_idx, min_x = polys[:,  ::2].argmin(1)
        max_y_idx, max_y = polys[:, 1::2].argmax(1)
        min_y_idx, min_y = polys[:, 1::2].argmin(1)

        hbboxes = jt.stack([min_x, min_y, max_x, max_y], dim=1)

        polys = polys.view(-1, 4, 2)
        num_polys = polys.size(0)
        polys_ordered = jt.zeros_like(polys)
        polys_ordered[:, 0] = polys[range(num_polys), min_y_idx]
        polys_ordered[:, 1] = polys[range(num_polys), max_x_idx]
        polys_ordered[:, 2] = polys[range(num_polys), max_y_idx]
        polys_ordered[:, 3] = polys[range(num_polys), min_x_idx]

        t_x = polys_ordered[:, 0, 0]
        r_y = polys_ordered[:, 1, 1]
        d_x = polys_ordered[:, 2, 0]
        l_y = polys_ordered[:, 3, 1]

        dt = (t_x - hbboxes[:, 0]) / (hbboxes[:, 2] - hbboxes[:, 0])
        dr = (r_y - hbboxes[:, 1]) / (hbboxes[:, 3] - hbboxes[:, 1])
        dd = (hbboxes[:, 2] - d_x) / (hbboxes[:, 2] - hbboxes[:, 0])
        dl = (hbboxes[:, 3] - l_y) / (hbboxes[:, 3] - hbboxes[:, 1])

        h_mask = (polys_ordered[:, 0, 1] - polys_ordered[:, 1, 1] == 0) | \
                (polys_ordered[:, 1, 0] - polys_ordered[:, 2, 0] == 0)
        fix_deltas = jt.stack([dt, dr, dd, dl], dim=1)
        fix_deltas[h_mask, :] = 1
        return fix_deltas

    def decode(self, hbboxes, fix_deltas):
        x1 = hbboxes[:, 0::4]
        y1 = hbboxes[:, 1::4]
        x2 = hbboxes[:, 2::4]
        y2 = hbboxes[:, 3::4]
        w = hbboxes[:, 2::4] - hbboxes[:, 0::4]
        h = hbboxes[:, 3::4] - hbboxes[:, 1::4]

        pred_t_x = x1 + w * fix_deltas[:, 0::4]
        pred_r_y = y1 + h * fix_deltas[:, 1::4]
        pred_d_x = x2 - w * fix_deltas[:, 2::4]
        pred_l_y = y2 - h * fix_deltas[:, 3::4]

        polys = jt.stack([pred_t_x, y1,
                             x2, pred_r_y,
                             pred_d_x, y2,
                             x1, pred_l_y], dim=-1)
        polys = polys.flatten(1)
        return polys

    
@BOXES.register_module()
class GVRatioCoder:
    def __init__(self):
        pass

    def encode(self, polys):
        assert polys.size(1) == 8
        hbboxes = poly2hbb(polys)
        h_areas = (hbboxes[:, 2] - hbboxes[:, 0]) * \
                (hbboxes[:, 3] - hbboxes[:, 1])

        polys = polys.view(polys.size(0), 4, 2)

        areas = jt.zeros(polys.size(0), dtype=polys.dtype)
        for i in range(4):
            areas += 0.5 * (polys[:, i, 0] * polys[:, (i+1)%4, 1] -
                            polys[:, (i+1)%4, 0] * polys[:, i, 1])
        areas = jt.abs(areas)

        ratios = areas / h_areas
        return ratios[:, None]

    def decode(self, bboxes, bboxes_pred):
        raise NotImplementedError

@BOXES.register_module()
class GVDeltaXYWHBBoxCoder:

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.)):
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
        assert bboxes.size() == gt_bboxes.size()

        pred_bboxes = bboxes.float()
        gt = gt_bboxes.float()
        px = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) * 0.5
        py = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) * 0.5
        pw = pred_bboxes[..., 2] - pred_bboxes[..., 0]
        ph = pred_bboxes[..., 3] - pred_bboxes[..., 1]

        gx = (gt[..., 0] + gt[..., 2]) * 0.5
        gy = (gt[..., 1] + gt[..., 3]) * 0.5
        gw = gt[..., 2] - gt[..., 0]
        gh = gt[..., 3] - gt[..., 1]

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = jt.log(gw / pw)
        dh = jt.log(gh / ph)
        deltas = jt.stack([dx, dy, dw, dh], dim=-1)

        means = jt.array(self.means, dtype=deltas.dtype).unsqueeze(0)
        stds = jt.array(self.stds, dtype=deltas.dtype).unsqueeze(0)
        deltas = (deltas - means) / stds

        return deltas

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):

        assert pred_bboxes.size(0) == bboxes.size(0)

        means = jt.array(self.means, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 4)
        stds = jt.array(self.stds, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 4)
        denorm_deltas = pred_bboxes * stds + means

        dx = denorm_deltas[:, 0::4]
        dy = denorm_deltas[:, 1::4]
        dw = denorm_deltas[:, 2::4]
        dh = denorm_deltas[:, 3::4]
        
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
        dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)

        # Compute center of each roi
        px = ((bboxes[:, 0] + bboxes[:, 2]) * 0.5).unsqueeze(1)
        py = ((bboxes[:, 1] + bboxes[:, 3]) * 0.5).unsqueeze(1)
        # Compute width/height of each roi
        pw = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(1)
        ph = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(1)

        # Use exp(network energy) to enlarge/shrink each roi
        gw = pw * dw.exp()
        gh = ph * dh.exp()
        # Use network energy to shift the center of each roi
        gx = px + pw * dx
        gy = py + ph * dy
        # Convert center-xy/width/height to top-left, bottom-right
        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5
        
        if max_shape is not None:
            x1 = x1.clamp(min_v=0, max_v=max_shape[1])
            y1 = y1.clamp(min_v=0, max_v=max_shape[0])
            x2 = x2.clamp(min_v=0, max_v=max_shape[1])
            y2 = y2.clamp(min_v=0, max_v=max_shape[0])

        bboxes = jt.stack([x1, y1, x2, y2], dim=-1).view_as(pred_bboxes)

        return bboxes

@BOXES.register_module()
class MidpointOffsetCoder:

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1., 1.)):
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):

        assert bboxes.size(0) == gt_bboxes.size(0)

        pred_bboxes = bboxes.float()
        gt = gt_bboxes.float()

        px = (pred_bboxes[..., 0] + pred_bboxes[..., 2]) * 0.5
        py = (pred_bboxes[..., 1] + pred_bboxes[..., 3]) * 0.5
        pw = pred_bboxes[..., 2] - pred_bboxes[..., 0]
        ph = pred_bboxes[..., 3] - pred_bboxes[..., 1]

        hbb, poly = obb2hbb(gt), obb2poly(gt)

        gx = (hbb[..., 0] + hbb[..., 2]) * 0.5
        gy = (hbb[..., 1] + hbb[..., 3]) * 0.5
        gw = hbb[..., 2] - hbb[..., 0]
        gh = hbb[..., 3] - hbb[..., 1]

        x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]

        _, y_min = y_coor.argmin(dim=1, keepdims=True)
        _, x_max = x_coor.argmax(dim=1, keepdims=True)

        _x_coor = x_coor.clone()
        _x_coor[jt.abs(y_coor-y_min) > 0.1] = -1000
        _, ga = _x_coor.argmax(1)

        _y_coor = y_coor.clone()
        _y_coor[jt.abs(x_coor-x_max) > 0.1] = -1000
        _, gb = _y_coor.argmax(1)

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = jt.log(gw / pw)
        dh = jt.log(gh / ph)
        da = (ga - gx) / gw
        db = (gb - gy) / gh
        deltas = jt.stack([dx, dy, dw, dh, da, db], dim=-1)

        means = jt.array(self.means, dtype=deltas.dtype).unsqueeze(0)
        stds = jt.array(self.stds, dtype=deltas.dtype).unsqueeze(0)
        deltas = (deltas - means) / stds

        return deltas

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):

        assert pred_bboxes.size(0) == bboxes.size(0)

        means = jt.array(self.means, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 6)
        stds = jt.array(self.stds, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 6)
        denorm_deltas = pred_bboxes * stds + means

        dx = denorm_deltas[:, 0::6]
        dy = denorm_deltas[:, 1::6]
        dw = denorm_deltas[:, 2::6]
        dh = denorm_deltas[:, 3::6]
        da = denorm_deltas[:, 4::6]
        db = denorm_deltas[:, 5::6]
        
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
        dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)
        
        # Compute center of each roi
        px = ((bboxes[:, 0] + bboxes[:, 2]) * 0.5).unsqueeze(1)     
        py = ((bboxes[:, 1] + bboxes[:, 3]) * 0.5).unsqueeze(1)
        
        # Compute width/height of each roi
        pw = (bboxes[:, 2] - bboxes[:, 0]).unsqueeze(1)
        ph = (bboxes[:, 3] - bboxes[:, 1]).unsqueeze(1)
        
        # Use exp(network energy) to enlarge/shrink each roi
        gw = pw * dw.exp()
        gh = ph * dh.exp()

        # Use network energy to shift the center of each roi
        gx = px + pw * dx
        gy = py + ph * dy

        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5

        da = da.clamp(min_v=-0.5, max_v=0.5)
        db = db.clamp(min_v=-0.5, max_v=0.5)
        ga = gx + da * gw
        _ga = gx - da * gw
        gb = gy + db * gh
        _gb = gy - db * gh

        polys = jt.stack([ga, y1, x2, gb, _ga, y2, x1, _gb], dim=-1)
        center = jt.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
        center_polys = polys - center
        diag_len = jt.sqrt(center_polys[..., 0::2] ** 2 + center_polys[..., 1::2] ** 2)
        _, max_diag_len = diag_len.argmax(dim=-1, keepdims=True)
        diag_scale_factor = max_diag_len / diag_len
        center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=-1)
        rectpolys = center_polys + center
        obboxes = rectpoly2obb(rectpolys).flatten(-2)
        return obboxes

@BOXES.register_module()
class OrientedDeltaXYWHTCoder:

    def __init__(self,
                 target_means=(0., 0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1., 1.)):
        self.means = target_means
        self.stds = target_stds

    def encode(self, bboxes, gt_bboxes):

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 5

        pred_bboxes = bboxes.float()
        gt = gt_bboxes.float()
        px, py, pw, ph, ptheta = pred_bboxes.unbind(dim=-1)
        gx, gy, gw, gh, gtheta = gt.unbind(dim=-1)

        dtheta1 = regular_theta(gtheta - ptheta)
        dtheta2 = regular_theta(gtheta - ptheta + np.pi/2)
        abs_dtheta1 = jt.abs(dtheta1)
        abs_dtheta2 = jt.abs(dtheta2)

        gw_regular = gw * (abs_dtheta1 < abs_dtheta2) + gh * (1 - (abs_dtheta1 < abs_dtheta2))
        gh_regular = gh * (abs_dtheta1 < abs_dtheta2) + gw * (1 - (abs_dtheta1 < abs_dtheta2))
        dtheta = dtheta1 * (abs_dtheta1 < abs_dtheta2) + dtheta2 * (1 - (abs_dtheta1 < abs_dtheta2))
        # gw_regular = jt.where(abs_dtheta1 < abs_dtheta2, gw, gh)
        # gh_regular = jt.where(abs_dtheta1 < abs_dtheta2, gh, gw)
        # dtheta = jt.where(abs_dtheta1 < abs_dtheta2, dtheta1, dtheta2)
        dx = (jt.cos(-ptheta) * (gx - px) + jt.sin(-ptheta) * (gy - py)) / pw
        dy = (-jt.sin(-ptheta) * (gx - px) + jt.cos(-ptheta) * (gy - py)) / ph
        dw = jt.log(gw_regular / pw)
        dh = jt.log(gh_regular / ph)
        deltas = jt.stack([dx, dy, dw, dh, dtheta], dim=-1)

        means = jt.array(self.means, dtype=deltas.dtype).unsqueeze(0)
        stds = jt.array(self.stds, dtype=deltas.dtype).unsqueeze(0)
        deltas = (deltas - means) / stds

        return deltas

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):

        assert pred_bboxes.size(0) == bboxes.size(0)

        means = jt.array(self.means, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 5)
        stds = jt.array(self.stds, dtype=pred_bboxes.dtype).repeat(1, pred_bboxes.size(1) // 5)
        denorm_deltas = pred_bboxes * stds + means
        
        dx = denorm_deltas[:, 0::5]
        dy = denorm_deltas[:, 1::5]
        dw = denorm_deltas[:, 2::5]
        dh = denorm_deltas[:, 3::5]
        dtheta = denorm_deltas[:, 4::5]
        max_ratio = np.abs(np.log(wh_ratio_clip))
        dw = dw.clamp(min_v=-max_ratio, max_v=max_ratio)
        dh = dh.clamp(min_v=-max_ratio, max_v=max_ratio)

        px, py, pw, ph, ptheta = bboxes.unbind(dim=-1)

        px = px.unsqueeze(1).expand_as(dx)
        py = py.unsqueeze(1).expand_as(dy)
        pw = pw.unsqueeze(1).expand_as(dw)
        ph = ph.unsqueeze(1).expand_as(dh)
        ptheta = ptheta.unsqueeze(1).expand_as(dtheta)

        gx = dx * pw * jt.cos(-ptheta) - dy * ph * jt.sin(-ptheta) + px
        gy = dx * pw * jt.sin(-ptheta) + dy * ph * jt.cos(-ptheta) + py
        gw = pw * dw.exp()
        gh = ph * dh.exp()
        gtheta = regular_theta(dtheta + ptheta)

        new_bboxes = jt.stack([gx, gy, gw, gh, gtheta], dim=-1)
        new_bboxes = regular_obb(new_bboxes)
        return new_bboxes.view_as(pred_bboxes)