from calendar import c
import jittor as jt
import numpy as np
from jittor.misc import _pair
from jdet.utils.registry import BOXES

@BOXES.register_module()
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

@BOXES.register_module()
class AnchorGenerator:
    """Standard anchor generator for 2D anchor-based detectors.
    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.
    Examples:
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)])
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)])
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = jt.array(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = jt.array(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = jt.array(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(jt.array): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.
        Args:
            base_size (int | float): Basic size of an anchor.
            scales (jt.array): Scales of the anchor.
            ratios (jt.array): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            jt.array: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = jt.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = jt.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.
        Args:
            x (jt.array): Grids of x dimension.
            y (jt.array): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.
        Returns:
            tuple[jt.array]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
        Return:
            list[jt.array]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(featmap_sizes[i], level_idx=i)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self, featmap_size, level_idx):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method ``self.grid_priors``.
        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
        Returns:
            jt.array: Anchors in the overall feature maps.
        """

        base_anchors = self.base_anchors[level_idx]
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = jt.arange(0, feat_w) * stride_w
        shift_y = jt.arange(0, feat_h) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = jt.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype="float32"):
        """Generate sparse anchors according to the ``prior_idxs``.
        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (h, w).
            level_idx (int): The level index of corresponding feature
                map.
        Returns:
            Tensor: Anchor with shape (N, 4), N should be equal to
                the length of ``prior_idxs``.
        """

        height, width = featmap_size
        num_base_anchors = self.num_base_anchors[level_idx]
        base_anchor_id = prior_idxs % num_base_anchors
        x = (prior_idxs //
             num_base_anchors) % width * self.strides[level_idx][0]
        y = (prior_idxs // width //
             num_base_anchors) % height * self.strides[level_idx][1]
        priors = jt.stack([x, y, x, y], 1).to(dtype) + \
            self.base_anchors[level_idx][base_anchor_id, :]

        return priors

    def grid_anchors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
        Return:
            list[jt.array]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method ``self.grid_anchors``.
        Args:
            base_anchors (jt.array): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
        Returns:
            jt.array: Anchors in the overall feature maps.
        """

        # keep featmap_size as Tensor instead of int, so that we
        # can covert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = jt.arange(0, feat_w,) * stride[0]
        shift_y = jt.arange(0, feat_h) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = jt.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape):
        """Generate valid flags of anchors in multiple feature levels.
        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
        Return:
            list(jt.array): Valid flags of anchors in multiple levels.
        """
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

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors):
        """Generate the valid flags of anchor in a single feature map.
        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
        Returns:
            jt.array: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = jt.zeros(feat_w).bool()
        valid_y = jt.zeros(feat_h).bool()
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0),num_base_anchors).view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str



@BOXES.register_module()
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
        valid = valid[:, None].expand((
            valid.size(0), num_base_anchors)).view(-1)
        return valid


@BOXES.register_module()
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

@BOXES.register_module()
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
