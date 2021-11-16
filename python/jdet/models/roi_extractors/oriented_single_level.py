import jittor as jt
import jittor.nn as nn

from jdet.ops import roi_align_rotated_v1
from jdet.utils.registry import ROI_EXTRACTORS
from jittor.misc import _pair

@ROI_EXTRACTORS.register_module()
class OrientedSingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 extend_factor=(1., 1.),
                 finest_scale=56):
        super(OrientedSingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.extend_factor = extend_factor
        self.finest_scale = finest_scale

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')

        assert hasattr(roi_align_rotated_v1, layer_type)
        layer_cls = getattr(roi_align_rotated_v1, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = jt.sqrt(rois[:, 3] * rois[:, 4])
        target_lvls = jt.floor(jt.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min_v=0, max_v=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 6)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """
        if scale_factor is None:
            return rois
        h_scale_factor, w_scale_factor = _pair(scale_factor)
        new_rois = rois.clone()
        new_rois[:, 3] = w_scale_factor * new_rois[:, 3]
        new_rois[:, 4] = h_scale_factor * new_rois[:, 4]
        return new_rois

    def execute(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size[0]           
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = jt.zeros(shape=(rois.shape[0], self.out_channels,
                                            out_size, out_size), dtype="float32")

        rois = self.roi_rescale(rois, self.extend_factor)
        target_lvls = self.map_roi_levels(rois, num_levels)
        rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any_():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
            else:
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.

        return roi_feats
