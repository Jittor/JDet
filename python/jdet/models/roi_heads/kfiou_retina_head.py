import numpy as np
from jittor import nn

from jdet.ops import box_iou_rotated
from jdet.utils.registry import build_from_cfg,HEADS,BOXES,LOSSES
from jdet.models.boxes.box_ops import bbox2loc, bbox_iou, loc2bbox, loc2bbox_r, bbox2loc_r
from jdet.models.boxes.box_ops import rotated_box_to_bbox, boxes_xywh_to_x0y0x1y1, boxes_x0y0x1y1_to_xywh, rotated_box_to_poly
from .retina_head import RetinaHead
# from  import get_var

@HEADS.register_module()
class KFIoURetinaHead(RetinaHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    The difference from `RetinaHead` is that its loss_bbox requires bbox_pred,
    bbox_targets, pred_decode and targets_decode as inputs.
    """

    #TODO: check 'H' mode
    def __init__(self,
                 n_class,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.4,
                 neg_iou_thresh_lo=0.,
                 nms_pre = 1000,
                 max_dets = 100,
                 anchor_generator=None,
                 bbox_coder = None,
                 mode='H',
                 score_threshold = 0.05,
                 nms_iou_threshold = 0.5,
                 roi_beta = 0.,
                 loc_loss = None,
                 cls_loss = None,
                 cls_loss_weight=1.,
                 loc_loss_weight=0.2,
                 ):
        super(KFIoURetinaHead, self).__init__()

        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.stacked_convs = stacked_convs
        self.nms_pre = nms_pre
        self.max_dets = max_dets
        self.mode = mode
        
        self.anchor_generator = build_from_cfg(anchor_generator, BOXES)
        self.bbox_coder = build_from_cfg(bbox_coder, BOXES)
        self.loc_loss = build_from_cfg(loc_loss, LOSSES)
        self.cls_loss = build_from_cfg(cls_loss, LOSSES)
        self.anchor_mode = anchor_generator.mode
        n_anchor = self.anchor_generator.num_base_anchors[0]

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))
            self.reg_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))
        
        self.n_class = n_class
        self.retina_cls = nn.Conv(feat_channels,n_anchor * n_class,3,padding=1)
        if (self.mode == 'H'):
            self.retina_reg = nn.Conv(feat_channels, n_anchor * 4, 3, padding=1)
        else:
            self.retina_reg = nn.Conv(feat_channels, n_anchor * 5, 3, padding=1)

        self.roi_beta = roi_beta
        self.nms_thresh = nms_iou_threshold
        self.score_thresh = score_threshold
        self.cls_loss_weight = cls_loss_weight
        self.loc_loss_weight = loc_loss_weight

        self.init_weights()

    def losses(self, all_proposals_, all_bbox_pred_, all_cls_score_, all_gt_roi_locs_, all_gt_roi_labels_):
        """Compute KFIoURetinanet loss:

        Args:
            all_proposals_ (List[jittor.Var]): basic anchors, shape (N, num_total_anchors, 5)
            all_bbox_pred_ (List[jittor.Var]): offset preds, shape (N, num_total_anchors, 5)
        """
        batch_size = len(all_bbox_pred_)
        losses = dict(
            roi_cls_loss = 0,
            roi_loc_loss = 0
        )

        for i in range(batch_size):
            # decode to boxes
            all_proposals_[i] = boxes_x0y0x1y1_to_xywh(all_proposals_[i])
            all_proposals_[i] = self.cvt2_w_greater_than_h(all_proposals_[i])
            all_proposals_[:, 4] += 0.5 * np.pi
            bbox_pred_deocde_ = loc2bbox_r(all_proposals_[i], all_bbox_pred_[i])
            gt_roi_locs_decode_ = loc2bbox_r(all_proposals_[i], all_gt_roi_locs_[i])

            all_gt_roi_labels = all_gt_roi_labels_[i]
            normalizer = max((all_gt_roi_labels>0).sum().item(),1)

            # regression loss
            roi_loc_loss = self.loc_loss(pred=all_bbox_pred_[i],
                                         target=all_gt_roi_locs_[i],
                                         pred_decode=bbox_pred_deocde_,
                                         targets_decode=gt_roi_locs_decode_,
                                         weight=all_gt_roi_labels)

            # classification loss
            inputs = all_cls_score_[i][all_gt_roi_labels >= 0]
            cates = all_gt_roi_labels[all_gt_roi_labels >= 0]#.unsqueeze(1)
            roi_cls_loss = self.cls_loss(inputs,cates,reduction_override = "sum")

            losses['roi_cls_loss'] += roi_cls_loss/normalizer
            losses['roi_loc_loss'] += roi_loc_loss/normalizer

        losses['roi_cls_loss'] *= self.cls_loss_weight / batch_size
        losses['roi_loc_loss'] *= self.loc_loss_weight / batch_size
        return losses
        