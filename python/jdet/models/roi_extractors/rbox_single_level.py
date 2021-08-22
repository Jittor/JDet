import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import ROI_EXTRACTORS
from jdet.ops import roi_align_rotated, roi_align

#TODO: replace with SingleRoIExtractor
@ROI_EXTRACTORS.register_module()
class RboxSingleRoIExtractor(nn.Module):
    """Extract RRoI features from a single level feature map.

    If there are mulitple input feature levels, each RRoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RRoI layer type and arguments.
        out_channels (int): Output channels of RRoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """
    def __init__(self,
                roi_layer,
                out_channels,
                featmap_strides,
                finest_scale=56,
                w_enlarge=1.2,
                h_enlarge=1.4):
        super(RboxSingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.w_enlarge = w_enlarge
        self.h_enlarge = h_enlarge

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)
    
    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(roi_align_rotated, layer_type)
        layer_cls = getattr(roi_align_rotated, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers
    
    def map_roi_levels(self, rois, num_levels):
        """Map rrois to corresponding feature levels by scales.

        - scale < finest_scale: level 0
        - finest_scale <= scale < finest_scale * 2: level 1
        - finest_scale * 2 <= scale < finest_scale * 4: level 2
        - scale >= finest_scale * 4: level 3

        Args:
            rois (Tensor): Input RRoIs, shape (k, 6). (index, x, y, w, h, angle)
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = jt.sqrt(rois[:, 3] * rois[:, 4])
        target_lvls = jt.floor(jt.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min_v=0, max_v=num_levels - 1).long()
        return target_lvls

    def execute(self, feats, rois):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].output_size[0]            #not sure
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        if isinstance(out_size, int):
            roi_feats = jt.zeros(shape=(rois.shape[0], self.out_channels,
                                    out_size, out_size), dtype="float32")
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            roi_feats = jt.zeros(shape=(rois.shape[0], self.out_channels,
                                    out_size[0], out_size[1]), dtype="float32")

        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any_():
                rois_ = rois[inds, :]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] += roi_feats_t
        return roi_feats