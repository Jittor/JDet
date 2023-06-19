from .anchor_target import images_to_levels
from .anchor_head import NewBaseAnchorHead
from .fake_rotated_retina_head import NewRotatedRetinaHead
from .s2anet_head import AlignConv

from jdet.utils.registry import HEADS
from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.ops.orn import ORConv2d, RotationInvariantPooling
from jdet.ops.dcn_v1 import DeformConv
from jdet.utils.general import multi_apply

import jittor as jt
from jittor import nn

def _get_refine_anchors(featmap_sizes, refine_anchors, targets):
    num_levels = len(featmap_sizes)
    refine_anchors_list = []
    for img_id in range(len(targets)):
        mlvl_refine_anchors = []
        for i in range(num_levels):
            refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
            mlvl_refine_anchors.append(refine_anchor)
        refine_anchors_list.append(mlvl_refine_anchors)
    return refine_anchors_list

class FeatureAlignmentModule(NewBaseAnchorHead):
    pass

class OrientedDetectionModule(NewBaseAnchorHead):
    def loss(self, refine_anchors, cls_scores, bbox_preds, targets):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        anchor_list = _get_refine_anchors(featmap_sizes, refine_anchors, targets)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(jt.contrib.concat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors)

        valid_flag_list = []
        for img_id, target in enumerate(targets):
            multi_level_flags = self.anchor_generator.valid_flags(featmap_sizes, target['pad_shape'][::-1])
            valid_flag_list.append(multi_level_flags)

        cls_reg_targets = self.get_targets(anchor_list, valid_flag_list, targets)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

@HEADS.register_module()
class NewS2ANetHead(NewRotatedRetinaHead):
    def __init__(self, *args, 
                stacked_convs=2,
                with_orconv=True,
                fam_cfg=dict(),
                odm_cfg=dict(),
                anchor_generator=None,
                bbox_coder=None,
                **kw):
        self.with_orconv=with_orconv
        # TODO: sampling and use_sigmoid_cls of fam should be the same as odm? 
        self.fam = FeatureAlignmentModule(*args, anchor_generator=None, **fam_cfg, **kw)
        self.odm = OrientedDetectionModule(*args, anchor_generator=None, **odm_cfg, **kw)
        super(NewS2ANetHead, self).__init__(*args,
                                             stacked_convs=stacked_convs,
                                             anchor_generator=anchor_generator,
                                             bbox_coder=bbox_coder,
                                             loss_cls=odm_cfg.get('loss_cls', None),
                                             **kw)
        self.fam.anchor_generator = self.anchor_generator
        self.odm.anchor_generator = self.anchor_generator
    
    def _init_layers(self):
        self.relu = nn.ReLU()
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))
            self.fam_cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))

        self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1)
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)

        self.align_conv = AlignConv(self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_orconv:
            self.or_conv = ORConv2d(
                self.feat_channels, int(self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)

        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1))
            self.odm_cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))

        self.odm_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.odm_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.fam_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.fam_cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fam_reg, std=0.01)
        normal_init(self.fam_cls, std=0.01, bias=bias_cls)

        self.align_conv.init_weights()

        normal_init(self.or_conv, std=0.01)
        for m in self.odm_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.odm_cls_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        normal_init(self.odm_reg, std=0.01)
    
    def __bbox_decode(self, bbox_preds, anchors):
        """
        Decode bboxes from deltas
        :param bbox_preds: [N,5,H,W]
        :param anchors: [H*W,5]
        :param means: mean value to decode bbox
        :param stds: std value to decode bbox
        :return: [N,H,W,5]
        """
        num_imgs, _, H, W = bbox_preds.shape
        bboxes_list = []
        for img_id in range(num_imgs):
            bbox_pred = bbox_preds[img_id]
            # bbox_pred.shape=[5,H,W]
            bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            bboxes = self.bbox_coder.decode(anchors, bbox_delta, wh_ratio_clip=1e-6)
            bboxes = bboxes.reshape(H, W, 5)
            bboxes_list.append(bboxes)
        return jt.stack(bboxes_list, dim=0)

    def forward_single(self, x, num_level):
        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat)
        fam_bbox_pred = self.fam_reg(fam_reg_feat)

        # only forward during training
        if self.is_training():
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat)
            fam_cls_score = self.fam_cls(fam_cls_feat)
        else:
            fam_cls_score = None
            
        stride = self.anchor_generator.strides[num_level]
        featmap_size = tuple(fam_bbox_pred.shape[-2:])
        base_anchors = self.anchor_generator.base_anchors[num_level]
        init_anchors = self.anchor_generator.single_level_grid_anchors(base_anchors, featmap_size, stride)
        
        refine_anchor = self.__bbox_decode(fam_bbox_pred.detach(), init_anchors)

        align_feat = self.align_conv(x, refine_anchor.clone(), stride[0])

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)
        odm_bbox_pred = self.odm_reg(odm_reg_feat)

        return fam_cls_score, fam_bbox_pred, refine_anchor, odm_cls_score, odm_bbox_pred
    
    def loss(self,
             fam_cls_scores,
             fam_bbox_preds,
             refine_anchors,
             odm_cls_scores,
             odm_bbox_preds,
             targets):
        fam_loss = self.fam.loss(fam_cls_scores, fam_bbox_preds, targets)
        odm_loss = self.odm.loss(refine_anchors, odm_cls_scores, odm_bbox_preds, targets)
        return dict(loss_fam_cls=fam_loss['loss_cls'],
                    loss_fam_bbox=fam_loss['loss_bbox'],
                    loss_odm_cls=odm_loss['loss_cls'],
                    loss_odm_bbox=odm_loss['loss_bbox'])

    def get_bboxes(self,
                   fam_cls_scores,
                   fam_bbox_preds,
                   refine_anchors,
                   odm_cls_scores,
                   odm_bbox_preds,
                   targets):
        assert len(odm_cls_scores) == len(odm_bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        num_levels = len(odm_cls_scores)
        multi_level_anchors = _get_refine_anchors(featmap_sizes, refine_anchors, targets)
        result_list = []
        for img_id, target in enumerate(targets):
            cls_score_list = [
                odm_cls_scores[i][img_id] for i in range(num_levels)
            ]
            bbox_pred_list = [
                odm_bbox_preds[i][img_id] for i in range(num_levels)
            ]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, multi_level_anchors[img_id], target)
            result_list.append(proposals)
        return result_list
    
    def execute_train(self, *args):
        return self.loss(*args)

    def execute_test(self, *args):
        return self.get_bboxes(*args)

    def execute(self, feats,targets):
        outs = multi_apply(self.forward_single, feats, range(self.anchor_generator.num_levels))
        if self.is_training():
            return self.execute_train(*outs,targets)
        else:
            return self.execute_test(*outs,targets)
