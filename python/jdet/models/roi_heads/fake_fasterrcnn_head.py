from .anchor_head import NewBaseAnchorHead
from jdet.utils.general import multi_apply
from jdet.ops.nms import nms_v0

from jdet.utils.registry import HEADS

from jittor import nn, init
import jittor as jt

@HEADS.register_module()
class NewFasterRCNNHead(NewBaseAnchorHead):
    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.anchor_generator.num_base_anchors[0] * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.anchor_generator.num_base_anchors[0] * 4, 1)
        self.init_weights()

    def init_weights(self):
        for var in [self.rpn_conv,self.rpn_cls, self.rpn_reg]:
            init.gauss_(var.weight,0,0.01)
            init.constant_(var.bias,0.0)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = nn.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           multi_level_anchors,
                           target,
                        ):
        cfg = self.cfg
        mlvl_proposals = []
        for idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[idx]
            rpn_bbox_pred = bbox_pred_list[idx]
            assert rpn_cls_score.shape[-2:] == rpn_bbox_pred.shape[-2:]
            anchors = multi_level_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if self.cfg['nms_pre'] > 0 and scores.shape[0] > cfg['nms_pre']:
                _, topk_inds = scores.topk(cfg['nms_pre'])
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # img_size:(w, h)
            proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, target['img_size'][::-1])
            if cfg['min_bbox_size'] > 0:
                w = proposals[:, 2] - proposals[:, 0]
                h = proposals[:, 3] - proposals[:, 1]
                valid_inds = jt.logical_and((w >= cfg['min_bbox_size']), (h >= cfg['min_bbox_size']))
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = jt.contrib.concat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals_inds = nms_v0(proposals, cfg['nms_thr'])
            proposals = proposals[proposals_inds]
            proposals = proposals[:cfg['nms_post'], :]
            mlvl_proposals.append(proposals)
        proposals = jt.contrib.concat(mlvl_proposals, 0)
        if cfg['nms_across_levels']:
            proposals_inds = nms_v0(proposals, cfg['nms_thr'])
            proposals = proposals[proposals_inds]
            proposals = proposals[:cfg['max_num'], :]
        else:
            scores = proposals[:, 4]
            num = min(cfg['max_num'], proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals

    def get_targets_parse(self, target):
        if self.bbox_type == 'hbb':
            if target["hboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["hboxes"].clone()

            if target["hboxes_ignore"] is None or target["hboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["hboxes_ignore"].clone()
        gt_labels = None
        # img_size:(w, h) img_shape:(h, w)
        img_shape = target["img_size"][::-1]
        return gt_bboxes, gt_bboxes_ignore, gt_labels, img_shape

    def execute(self,features,targets):

        outs = multi_apply(self.forward_single, features)
        if self.is_training():
            losses = self.loss(*outs,targets)
        else:
            losses = dict()
        proposals = self.get_bboxes(*outs,targets)
        return proposals, losses


