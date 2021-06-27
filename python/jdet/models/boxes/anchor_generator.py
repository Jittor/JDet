import jittor as jt 
import numpy as np

class AnchorGeneratorRotated:
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
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.ratios)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            ctr = None
            if self.ctr is not None:
                ctr = self.ctr[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    angles=self.angles,
                    ctr=ctr))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_size, scales, ratios, angles, ctr):
        w = base_size
        h = base_size
        if ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = ctr

        h_ratios = jt.sqrt(ratios)
        w_ratios = 1 / h_ratios
        assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        ws = (w * w_ratios[:, None, None] * scales[None, :, None] *
              jt.ones_like(angles)[None, None, :]).view(-1)
        hs = (h * h_ratios[:, None, None] * scales[None, :, None] *
              jt.ones_like(angles)[None, None, :]).view(-1)
        angles = angles.repeat(len(scales) * len(ratios))

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

    def single_level_grid_anchors(self, base_anchor, featmap_size, stride=16):
        # featmap_size*stride project it to original area

        feat_h, feat_w = featmap_size
        shift_x = jt.arange(0, feat_w) * stride[0]
        shift_y = jt.arange(0, feat_h) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_others = jt.zeros_like(shift_xx)
        shifts = jt.stack(
            [shift_xx, shift_yy, shift_others, shift_others, shift_others], dim=-1)
        shifts = shifts.cast(base_anchor.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 5) to K shifts (K, 1, 5) to get
        # shifted anchors (K, A, 5), reshape to (K*A, 5)

        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
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


class AnchorGenerator:
    """
    Examples:
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2))
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = jt.array(scales)
        self.ratios = jt.array(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        return len(self.ratios)

    def gen_base_anchors(self):
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            ctr = None
            if self.ctr is not None:
                ctr = self.ctr[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    ctr=ctr))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      ctr=None):
        w = base_size
        h = base_size
        if ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = ctr

        h_ratios = jt.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = jt.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

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

    def single_level_grid_anchors(self, base_anchor, featmap_size, stride=16):
        # featmap_size*stride project it to original area

        feat_h, feat_w = featmap_size
        shift_x = jt.arange(0, feat_w) * stride[0]
        shift_y = jt.arange(0, feat_h) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = jt.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.cast(base_anchor.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchor[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
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
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).view(-1)
        return valid


class SSDAnchorGenerator(AnchorGenerator):
    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True):
        assert len(strides) == len(ratios)

        self.strides = [(stride, stride) for stride in strides]
        self.input_size = input_size
        self.ctrs = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.basesize_ratio_range = basesize_ratio_range

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
                ctr=self.ctrs[i])
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
    anchor_generator = dict(
        scale_major=False,
        input_size=300,
        strides=[8, 16, 32, 64, 100, 300],
        ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        basesize_ratio_range=(0.15, 0.9)),
    anchor_generator = SSDAnchorGenerator(scale_major=False,
                                          input_size=300,
                                          strides=[8, 16, 32, 64, 100, 300],
                                          ratios=([2], [2, 3], [2, 3],
                                                  [2, 3], [2], [2]),
                                          basesize_ratio_range=(0.15, 0.9))
    anchor_bases = anchor_generator.base_anchors
    # feat_strides = anchor_generator['strides']
    print(anchor_bases)
