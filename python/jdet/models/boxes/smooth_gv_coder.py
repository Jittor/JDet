from jdet.utils.registry import BOXES
from jdet.models.boxes.box_ops import poly_to_rotated_box, rotated_box_to_poly

import jittor as jt

@BOXES.register_module()
class SmoothGVCoder:
    def __init__(self, pow_iou=1., ratio_ver=1, **kwargs):
        self.pow_iou = pow_iou
        self.ratio_ver = ratio_ver
        super(SmoothGVCoder, self).__init__(**kwargs)

    @jt.no_grad()
    def build_iou_matrix(self, hbboxes:jt.Var, ratio_pred:jt.Var):
        assert hbboxes.size(1) == 4
        assert ratio_pred.shape[0] == hbboxes.shape[0]
        min_x = hbboxes[:, 0]
        min_y = hbboxes[:, 1]
        max_x = hbboxes[:, 2]
        max_y = hbboxes[:, 3]
        w = max_x - min_x
        h = max_y - min_y
        w_large = w > h
        w_large_ratio = ratio_pred[w_large] / 4
        w_large_w = w[w_large]
        w_large_h = h[w_large]
        h_large = jt.logical_not(w_large)
        h_large_ratio = ratio_pred[h_large] / 4
        h_large_w = w[h_large]
        h_large_h = h[h_large]

        x_ratio = jt.zeros_like(ratio_pred)
        y_ratio = jt.zeros_like(ratio_pred)

        # x(1-x)=r --> x^2-x+r=0
        h_large_delta_x = jt.sqrt(1 - 4 * h_large_ratio)
        x_ratio[h_large] = (1 - h_large_delta_x) / 2
        # h^2y(1-y) = w^2r --> y^2-y+(w^2/h^2)r=0
        h_large_delta_y = jt.sqrt(1 - 4 * (h_large_w*h_large_w/(h_large_h*h_large_h)) * h_large_ratio)
        y_ratio[h_large] = (1 - h_large_delta_y) / 2
        
        # y(1-y)=r --> y^2-y+r=0
        w_large_delta_y = jt.sqrt(1 - 4 * w_large_ratio)
        y_ratio[w_large] = (1 - w_large_delta_y) / 2
        # w^2x(1-x) = h^2r --> x^2-x+(h^2/w^2)r=0
        w_large_delta_x = jt.sqrt(1 - 4 * (w_large_h*w_large_h/(w_large_w*w_large_w)) * w_large_ratio)
        x_ratio[w_large] = (1 - w_large_delta_x) / 2
        iou_self = jt.zeros_like(ratio_pred)

        # type0 shape: l01, l02
        # type1 shape: l03, l04
        l_01 = jt.sqrt(jt.sqr(x_ratio * w) + jt.sqr(y_ratio * h))
        l_02 = jt.sqrt(jt.sqr((1 - x_ratio) * w) + jt.sqr((1 - y_ratio) * h))
        l_03 = jt.sqrt(jt.sqr(x_ratio * w) + jt.sqr((1 - y_ratio) * h))
        l_04 = jt.sqrt(jt.sqr((1 - x_ratio) * w) + jt.sqr(y_ratio * h))
        # (yh)t/(xw) + (1-y)ht/(xw) = (1-2x)w --> t=(1-2x)xw^2/h
        # intersection = (1 - t / ((1-y)h)) * l02 * l01 = (1 - (1-2x)xw^2/((1-y)h^2)) * l01 * l02
        i_01 = (1 - (1 - 2 * x_ratio) * x_ratio * w * w / ((1 - y_ratio) * h * h)) * l_01 * l_02
        iou_01 = i_01 / (l_01 * l_02 + l_03 * l_04 - i_01)
        i_02 = (1 - (1 - 2 * y_ratio) * y_ratio * h * h / ((1 - x_ratio) * w * w)) * l_01 * l_02
        iou_02 = i_02 / (l_01 * l_02 + l_03 * l_04 - i_02)
        # i_03 = 2 * x_ratio * y_ratio * w * h
        i_03 = jt.sqr(x_ratio + y_ratio - 2 * x_ratio * y_ratio) / ((1 - x_ratio) * (1 - y_ratio)) * w * h / 2
        iou_03 = jt.zeros_like(iou_02)
        nzero = l_01 > 1e-5
        iou_03[nzero] = i_03[nzero] / (l_01[nzero] * l_02[nzero] * 2 - i_03[nzero])

        h1 = 0.5 * w - (0.5 - y_ratio) / (1 - y_ratio) * w * x_ratio
        h2 = 0.5 * h - (0.5 - x_ratio) / (1 - x_ratio) * h * y_ratio
        s2 = jt.sqr(h1) + jt.sqr(h2)
        tana = (0.5 - x_ratio) / (1 - x_ratio) * l_04 / (0.5 / (1 - y_ratio) * l_03)
        tanb = (0.5 - y_ratio) / (1 - y_ratio) * l_03 / (0.5 / (1 - x_ratio) * l_04)
        nzero = tana + tanb > 1e-8
        i_12_tpye1 = jt.zeros_like(i_03)
        i_12_tpye1[nzero] = tana[nzero] * tanb[nzero] / (tana[nzero] + tanb[nzero])
        i_12 = i_12_tpye1 * s2 * 2 + h1 * h2 * 2
        iou_12 = i_12 / (l_03 * l_04 * 2 - i_12)
        iou_self = jt.ones_like(ratio_pred)
        iou0 = jt.stack([iou_self, iou_01, iou_02, iou_03], dim=-1)
        iou1 = jt.stack([iou_01, iou_self, iou_12, iou_02], dim=-1)
        iou2 = jt.stack([iou_02, iou_12, iou_self, iou_01], dim=-1)
        iou3 = jt.stack([iou_03, iou_02, iou_01, iou_self], dim=-1)
        return jt.stack([iou0, iou1, iou2, iou3], dim=-2)

    def build_polypairs(self, hbboxes:jt.Var, ratio_pred:jt.Var):
        assert hbboxes.size(1) == 4
        assert ratio_pred.shape[0] == hbboxes.shape[0]
        min_x = hbboxes[:, 0]
        min_y = hbboxes[:, 1]
        max_x = hbboxes[:, 2]
        max_y = hbboxes[:, 3]
        w = max_x - min_x
        h = max_y - min_y
        w_large = w > h
        w_large_ratio = ratio_pred[w_large] / 4
        w_large_w = w[w_large]
        w_large_h = h[w_large]
        h_large = jt.logical_not(w_large)
        h_large_ratio = ratio_pred[h_large] / 4
        h_large_w = w[h_large]
        h_large_h = h[h_large]
        x1 = jt.zeros_like(ratio_pred)
        x2 = jt.zeros_like(ratio_pred)
        y1 = jt.zeros_like(ratio_pred)
        y2 = jt.zeros_like(ratio_pred)

        # x(1-x)=r --> x^2-x+r=0
        h_large_delta_x = jt.sqrt(1 - 4 * h_large_ratio)
        x1[h_large] = (1 - h_large_delta_x) / 2 * h_large_w
        x2[h_large] = (1 + h_large_delta_x) / 2 * h_large_w
        # h^2y(1-y) = w^2r --> y^2-y+(w^2/h^2)r=0
        h_large_delta_y = jt.sqrt(1 - 4 * (h_large_w*h_large_w/(h_large_h*h_large_h)) * h_large_ratio)
        y1[h_large] = (1 - h_large_delta_y) / 2 * h_large_h
        y2[h_large] = (1 + h_large_delta_y) / 2 * h_large_h
        
        # y(1-y)=r --> y^2-y+r=0
        w_large_delta_y = jt.sqrt(1 - 4 * w_large_ratio)
        y1[w_large] = (1 - w_large_delta_y) / 2 * w_large_h
        y2[w_large] = (1 + w_large_delta_y) / 2 * w_large_h
        # w^2x(1-x) = h^2r --> x^2-x+(h^2/w^2)r=0
        w_large_delta_x = jt.sqrt(1 - 4 * (w_large_h*w_large_h/(w_large_w*w_large_w)) * w_large_ratio)
        x1[w_large] = (1 - w_large_delta_x) / 2 * w_large_w
        x2[w_large] = (1 + w_large_delta_x) / 2 * w_large_w

        poly1 = jt.stack([min_x+x1, min_y,
                             max_x, min_y+y2,
                             max_x-x1, max_y,
                             min_x, max_y-y2], dim=-1)
        poly2 = jt.stack([min_x+x2, min_y,
                             max_x, min_y+y2,
                             max_x-x2, max_y,
                             min_x, max_y-y2], dim=-1)
        poly3 = jt.stack([min_x+x1, min_y,
                             max_x, min_y+y1,
                             max_x-x1, max_y,
                             min_x, max_y-y1], dim=-1)
        poly4 = jt.stack([min_x+x2, min_y,
                             max_x, min_y+y1,
                             max_x-x2, max_y,
                             min_x, max_y-y1], dim=-1)
        return [poly_to_rotated_box(poly1),
                poly_to_rotated_box(poly2),
                poly_to_rotated_box(poly3),
                poly_to_rotated_box(poly4)]

    def encode(self, rbboxes:jt.Var):
        assert rbboxes.size(1) == 5

        polys = rotated_box_to_poly(rbboxes)

        max_x_idx, max_x = jt.argmax(polys[:, ::2], dim=1)
        min_x_idx, min_x = jt.argmin(polys[:, ::2], dim=1)
        max_y_idx, max_y = jt.argmax(polys[:, 1::2], dim=1)
        min_y_idx, min_y = jt.argmin(polys[:, 1::2], dim=1)
        hbboxes = jt.stack([min_x, min_y, max_x, max_y], dim=1)

        polys = polys.view(-1, 4, 2)
        w = hbboxes[:, 2] - hbboxes[:, 0]
        h = hbboxes[:, 3] - hbboxes[:, 1]
        x_ind = jt.argsort(polys[:, :, 0], dim=1)[0]
        y_ind = jt.argsort(polys[:, :, 1], dim=1)[0]
        polys_x = polys[:, :, 0]
        polys_y = polys[:, :, 1]
        s_x = polys_x[(jt.arange(polys.shape[0]), x_ind[:, 1])]
        s_y = polys_y[(jt.arange(polys.shape[0]), y_ind[:, 1])]
        dx = (s_x - hbboxes[:, 0]) / w
        dy = (s_y - hbboxes[:, 1]) / h

        w_large = w > h
        h_large_dx = dx[jt.logical_not(w_large)]
        w_large_dy = dy[w_large]
        ratio = jt.zeros_like(max_x)
        ratio[jt.logical_not(w_large)] = h_large_dx * (1 - h_large_dx) * 4
        ratio[w_large] = w_large_dy * (1 - w_large_dy) * 4

        assert jt.all(ratio <= 1.0 + 1e-5)

        ious = self.build_iou_matrix(hbboxes, ratio)
        is_type13 = jt.logical_or(x_ind[:, 1] == y_ind[:, 2], x_ind[:, 1] == y_ind[:, 3])
        is_type23 = jt.logical_or(x_ind[:, 0] == y_ind[:, 2], x_ind[:, 0] == y_ind[:, 3])
        rtype = is_type23.long() * 2 + is_type13.long()
        ious = ious[(jt.arange(ious.shape[0]), rtype)]
        ious = jt.pow(ious, self.pow_iou)
        if self.ratio_ver == 1:
            pass
        elif self.ratio_ver == 2:
            ratio = 1 - jt.sqrt(1 - ratio)
        else:
            raise NotImplementedError
        return ratio[:, None], ious
    
    def decode(self, hbboxes:jt.Var, bboxes_pred:jt.Var, rotated_scores:jt.Var):
        assert rotated_scores.size(1) == 4
        assert hbboxes.size(0) == hbboxes.size(0) == rotated_scores.size(0)
        bboxes_pred = jt.squeeze(bboxes_pred, dim=-1)
        if self.ratio_ver == 1:
            pass
        elif self.ratio_ver == 2:
            bboxes_pred = 1 - jt.sqr(1 - bboxes_pred)
        else:
            raise NotImplementedError

        rbboxes_list = self.build_polypairs(hbboxes, bboxes_pred)
        rbboxes_proposals = jt.concat([rbboxes[:, None, :] for rbboxes in rbboxes_list], dim=1)
        rbboxes_proposals = rbboxes_proposals.reshape(-1, rbboxes_proposals.shape[-1])

        num_hbboxes = hbboxes.size(0)
        best_index = jt.argmax(rotated_scores, dim=-1, keepdims=False)[0] + \
            jt.arange(num_hbboxes, dtype=jt.int32) * 4

        assert best_index.max() < rbboxes_proposals.shape[0]
        max_scores1 = jt.max(rotated_scores, dim=-1)
        rotated_scores = jt.flatten(rotated_scores)
        max_scores2 = rotated_scores[best_index]
        assert (max_scores1 - max_scores2).abs().max() < 1e-5

        best_rbboxes = rbboxes_proposals[best_index, :]
        return best_rbboxes

