from calendar import c
import jittor as jt
import numpy as np
from jdet.utils.registry import MODELS

class AnchorGeneratorRotatedS2ANet:
    def __init__(self, base_size, scales, ratios, angles=[0,],scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = jt.array(scales)
        self.ratios = jt.array(ratios)
        self.angles = jt.array(angles)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = jt.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        ws = (w * w_ratios[:, None, None] * self.scales[None, :, None] *
              jt.ones_like(self.angles)[None, None, :]).view(-1)
        hs = (h * h_ratios[:, None, None] * self.scales[None, :, None] *
              jt.ones_like(self.angles)[None, None, :]).view(-1)
        angles = self.angles.repeat(len(self.scales) * len(self.ratios))

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        x_ctr += jt.zeros_like(ws)
        y_ctr += jt.zeros_like(ws)
        base_anchors = jt.stack(
            [x_ctr, y_ctr, ws, hs, angles], dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16):
        # featmap_size*stride project it to original area
        base_anchors = self.base_anchors

        feat_h, feat_w = featmap_size
        shift_x = jt.arange(0, feat_w) * stride
        shift_y = jt.arange(0, feat_h) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_others = jt.zeros_like(shift_xx)
        shifts = jt.stack(
            [shift_xx, shift_yy, shift_others, shift_others, shift_others], dim=-1)
        shifts = shifts.cast(base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = jt.zeros((feat_w,)).bool()
        valid_y = jt.zeros((feat_h,)).bool()
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand((valid.size(0), self.num_base_anchors)).view(-1)
        return valid


@MODELS.register_module()
class AnchorGeneratorRotated:
    def __init__(self, strides, ratios, scales, base_sizes=None, angles=[0, ], scale_major=True, centers=None, center_offset=0.5, mode='H'):
        self.ratios = jt.array(ratios)
        self.scales = jt.array(scales)
        self.strides = [(stride, stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert(mode in ['H', 'R'])
        self.mode = mode
        self.angles = jt.array(angles) if self.mode == 'R' else jt.array([0.])
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.strides)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            centers = None
            if self.centers is not None:
                centers = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    angles=self.angles,
                    centers=centers))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_size, scales, ratios, angles, centers):
        w = base_size
        h = base_size
        if centers is None:
            x_ctr = self.center_offset * w
            y_ctr = self.center_offset * h
        else:
            x_ctr, y_ctr = centers

        h_ratios = jt.sqrt(ratios)
        w_ratios = 1 / h_ratios
        # assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        if self.scale_major and self.mode == 'R':
            ws = (w * w_ratios[:, None, None] * scales[None, :, None] *
                  jt.ones_like(angles)[None, None, :]).view(-1)
            hs = (h * h_ratios[:, None, None] * scales[None, :, None] *
                  jt.ones_like(angles)[None, None, :]).view(-1)
        else:
            ws = (w * scales[:, None, None] * w_ratios[None, :, None] *
                  jt.ones_like(angles)[None, None, :]).view(-1)
            hs = (h * scales[:, None, None] * h_ratios[None, :, None] *
                  jt.ones_like(angles)[None, None, :]).view(-1)
        angles = angles.repeat(len(scales) * len(ratios))

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        x_ctr += jt.zeros_like(ws)
        y_ctr += jt.zeros_like(ws)
        if (self.mode == 'H'):
            base_anchors = jt.stack(
                [x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                 x_ctr + 0.5 * ws, y_ctr + 0.5 * hs], dim=-1)
        else:
            base_anchors = jt.stack(
                [x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                 x_ctr + 0.5 * ws, y_ctr + 0.5 * hs, angles], dim=-1)
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes):
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self, base_anchor, featmap_size, stride=(16, 16)):
        # featmap_size*stride project it to original area

        feat_h, feat_w = featmap_size
        shift_x = jt.arange(0, feat_w) * stride[0]
        shift_y = jt.arange(0, feat_h) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_others = jt.zeros_like(shift_xx)
        if (self.mode == 'H'):
            shifts = jt.stack(
                [shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        else:
            shifts = jt.stack(
                [shift_xx, shift_yy, shift_xx, shift_yy, shift_others], dim=-1)
        shifts = shifts.cast(base_anchor.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
        if (self.mode == 'H'):
            all_anchors = all_anchors.view(-1, 4)
        else:
            all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape):

        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i])
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self, featmap_size, valid_size, num_base_anchors):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = jt.zeros((feat_w,)).bool()
        valid_y = jt.zeros((feat_h,)).bool()
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), num_base_anchors).view(-1)
        return valid


@MODELS.register_module()
class AnchorGeneratorYangXue(AnchorGeneratorRotated):
    def __init__(self, yx_base_size, **kwargs):
        # TODO: do not use yx_base_size
        self.yx_base_size = yx_base_size
        super(AnchorGeneratorYangXue, self).__init__(**kwargs)

    def gen_single_level_base_anchors(self, base_size, scales, ratios, angles, centers):
        w = base_size
        h = base_size
        if centers is None:
            x_ctr = self.center_offset * (self.yx_base_size - 1)
            y_ctr = self.center_offset * (self.yx_base_size - 1)
        else:
            x_ctr, y_ctr = centers

        h_ratios = jt.sqrt(ratios)
        w_ratios = 1 / h_ratios
        assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        ws = np.round(w * w_ratios[:, None, None] /
                      base_size * self.yx_base_size)
        hs = np.round(ws * ratios[:, None, None])
        ws = (ws / self.yx_base_size * base_size * scales[None, :, None] *
              jt.ones_like(angles)[None, None, :]).view(-1)
        # hs = np.round(h * h_ratios[:, None, None])
        hs = (hs / self.yx_base_size * base_size * scales[None, :, None] *
              jt.ones_like(angles)[None, None, :]).view(-1)
        angles = angles.repeat(len(scales) * len(ratios))

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        x_ctr += jt.zeros_like(ws)
        y_ctr += jt.zeros_like(ws)
        if (self.mode == 'H'):
            base_anchors = jt.stack(
                [x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                 x_ctr + 0.5 * ws, y_ctr + 0.5 * hs], dim=-1)
        else:
            base_anchors = jt.stack(
                [x_ctr - 0.5 * ws, y_ctr - 0.5 * hs,
                 x_ctr + 0.5 * ws, y_ctr + 0.5 * hs, angles], dim=-1)
        return base_anchors


class SSDAnchorGenerator(AnchorGeneratorRotated):
    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True,
                 mode='H'):
        assert len(strides) == len(ratios)

        self.strides = [(stride, stride) for stride in strides]
        self.input_size = input_size
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.basesize_ratio_range = basesize_ratio_range
        self.mode = mode

        # calculate anchor ratios and sizes
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
        min_sizes = []
        max_sizes = []
        for ratio in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(self.input_size * ratio / 100))
            max_sizes.append(int(self.input_size * (ratio + step) / 100))
        if self.input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(self.input_size * 10 / 100))
                max_sizes.insert(0, int(self.input_size * 20 / 100))
            else:
                raise ValueError(
                    'basesize_ratio_range[0] should be either 0.15'
                    'or 0.2 when input_size is 300, got '
                    f'{basesize_ratio_range[0]}.')
        elif self.input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(self.input_size * 4 / 100))
                max_sizes.insert(0, int(self.input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            else:
                raise ValueError('basesize_ratio_range[0] should be either 0.1'
                                 'or 0.15 when input_size is 512, got'
                                 f' {basesize_ratio_range[0]}.')
        else:
            raise ValueError('Only support 300 or 512 in SSDAnchorGenerator'
                             f', got {self.input_size}.')

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(jt.array(anchor_ratio))
            anchor_scales.append(jt.array(scales))

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.ctr_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                angles=jt.array([0, ]),
                centers=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = base_anchors[indices]
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str


if __name__ == '__main__':
    anchor_generator_rotated = AnchorGeneratorRotated(strides=[16], base_sizes=[
        9], scales=[1], ratios=[1])
    anchor_bases = anchor_generator_rotated.base_anchors
    print(anchor_bases)
    print(anchor_generator_rotated.grid_anchors([(2, 2)]))
    print('anchor_generator_rotated')
    # anchor_generator = AnchorGenerator(strides=[16], base_sizes=[
    #                                    9], scales=[1], ratios=[1])
    # anchor_bases = anchor_generator.base_anchors
    # print(anchor_bases)
    # print(anchor_generator.grid_anchors([(2, 2)]))
    # print('anchor_generator')

    ssd_anchor_generator = SSDAnchorGenerator(
        scale_major=False,
        input_size=300,
        strides=[8],
        ratios=([2],),
        basesize_ratio_range=(0.15, 0.9))
    anchor_bases = ssd_anchor_generator.base_anchors
    print(anchor_bases)
    print(ssd_anchor_generator.grid_anchors([(2, 2)]))
    print('ssd_anchor_generator')
