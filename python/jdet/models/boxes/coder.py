from .box_ops import delta2bbox,bbox2delta,delta2bbox_rotated,bbox2delta_rotated
from jdet.utils.registry import BOXES
from jdet.models.utils.gliding_transforms import *

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
            bboxes (jt.Var): source boxes, e.g., object proposals.
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
            bboxes (jt.Var): Source boxes, e.g., object proposals.
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
class DeltaXYWHTBBoxCoder:
    """Delta XYWHT BBox coder, for Retinanet.
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
            bboxes (jt.Var): Source boxes, e.g., object proposals.
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
        decoded_bboxes = self.delta2bbox_rotated_new(bboxes, pred_bboxes, self.means, self.stds,
                                            max_shape, wh_ratio_clip, self.clip_border)

        return decoded_bboxes

    def delta2bbox_rotated_new(self, rois, deltas, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), max_shape=None,
                       wh_ratio_clip=16 / 1000, clip_border=True):
        """
        Following OBBDetection
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
        gx = dx * roi_w * jt.cos(-roi_angle) \
            - dy * roi_h * jt.sin(-roi_angle) + roi_x
        gy = dx * roi_w * jt.sin(-roi_angle) \
            + dy * roi_h * jt.cos(-roi_angle) + roi_y
        gw = roi_w * dw.exp()
        gh = roi_h * dh.exp()

        # ga = np.pi * dangle + roi_angle

        ga = dangle + roi_angle
        # ga = self.norm_angle(ga)

        bboxes = jt.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
        bboxes = self.regular_obb(bboxes)
        return bboxes

    def regular_obb(self, obboxes):
        x, y, w, h, theta = obboxes.unbind(dim=-1)
        w_regular, h_regular, theta_regular = w.copy(), h.copy(), theta.copy()
        regular_index = jt.where(w <= h)
        for index in regular_index:
            w_regular[index] = h[index]
            h_regular[index] = w[index]
            theta_regular[index] += np.pi / 2
        theta_regular = regular_theta(theta_regular)
        return jt.stack([x, y, w_regular, h_regular, theta_regular], dim=-1)


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

        proposals = bboxes.float()
        gt = gt_bboxes.float()
        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]

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
