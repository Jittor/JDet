from .anchor_head import NewBaseAnchorHead
from jdet.utils.registry import HEADS
from jdet.ops.bbox_transforms import obb2hbb, hbb2obb
from jdet.ops.nms import nms_v0

from jittor import nn
from jittor import init
import jittor as jt

@HEADS.register_module()
class NewOrientedRPNHead(NewBaseAnchorHead):
    def _init_layers(self):
        """Initialize layers of the head."""
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)
        self.init_weights()
    
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
        elif self.bbox_type == 'obb':
            if target["rboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = target["rboxes"].clone()
                # gt_bboxes[:, -1] *= -1

            if target["rboxes_ignore"] is None or target["rboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = target["rboxes_ignore"].clone()
                # gt_bboxes_ignore[:, -1] *= -1
        elif self.bbox_type == 'hbb_as_obb':
            if target["hboxes"] is None:
                gt_bboxes = None
            else:
                gt_bboxes = hbb2obb(target["hboxes"].clone())

            if target["hboxes_ignore"] is None or target["hboxes_ignore"].numel() == 0: 
                gt_bboxes_ignore = None
            else:
                gt_bboxes_ignore = hbb2obb(target["hboxes_ignore"].clone())
        gt_labels = None
        # img_size:(w, h) img_shape:(h, w)
        img_shape = target["img_size"][::-1]
        return gt_bboxes, gt_bboxes_ignore, gt_labels, img_shape

    def init_weights(self):
        for var in [self.rpn_conv,self.rpn_cls, self.rpn_reg]:
            init.gauss_(var.weight,0,0.01)
            init.constant_(var.bias,0.0)

    def forward_single(self, x):
        """Forward feature map of a single scale level."""
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
        level_ids = []
        mlvl_scores = []
        mlvl_valid_anchors = []
        mlvl_bbox_pred = []
        for idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[idx]
            rpn_bbox_pred = bbox_pred_list[idx]
            assert rpn_cls_score.shape[-2:] == rpn_bbox_pred.shape[-2:]

            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, self.cls_out_channels)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]

            anchors = multi_level_anchors[idx]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, self.reg_dim)
            if cfg['nms_pre'] > 0 and scores.shape[0] > cfg['nms_pre']:
                _, topk_inds = scores.topk(cfg['nms_pre'])
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]

            mlvl_scores.append(scores)
            mlvl_bbox_pred.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(jt.full((scores.size(0), ), idx).long())

        anchors = jt.concat(mlvl_valid_anchors)
        rpn_bbox_pred = jt.concat(mlvl_bbox_pred)
        scores = jt.concat(mlvl_scores)

        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=target['img_size'][::-1])
        ids = jt.concat(level_ids)

        if cfg['min_bbox_size'] > 0:
            w, h = proposals[:, 2], proposals[:, 3]
            valid_mask = (w > cfg['min_bbox_size']) & (h > cfg['min_bbox_size'])
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]
        
        hproposals = obb2hbb(proposals)
        max_coordinate = hproposals.max() - hproposals.min()
        offsets = ids.astype(hproposals.dtype) * (max_coordinate + 1)
        hproposals += offsets[:, None]

        hproposals_concat = jt.concat([hproposals, scores.unsqueeze(1)], dim=1)
        keep = nms_v0(hproposals_concat, cfg['nms_thresh'])
        
        dets = jt.concat([proposals, scores.unsqueeze(1)], dim=1)
        dets = dets[keep, :]

        dets = dets[:cfg['nms_post']]

        return dets

    def execute_train(self, *args, **kwargs):
        proposals = self.get_bboxes(*args, **kwargs)
        losses = self.loss(*args, **kwargs)
        rpn_losses = dict(
            loss_rpn_cls = losses['loss_cls'],
            loss_rpn_bbox = losses['loss_bbox']
        )
        return proposals, rpn_losses
    
    def execute_test(self, cls_scores, bbox_preds, targets):
        return super().execute_test(cls_scores, bbox_preds, targets), None
