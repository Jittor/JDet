from jdet.models.losses.focal_loss import sigmoid_focal_loss
from jdet.models.losses.faster_rcnn_loss import faster_rcnn_loss
from jdet.models.roi_heads.anchor_generator import generate_anchor_base, grid_anchors
from jdet.utils.registry import ROI_HEADS
from jittor import nn 
import jittor as jt


@ROI_HEADS.register_module()
class RetinaHead(nn.Module):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 ratios=[0.5, 1.0, 2.0],
                 feat_strides=[8, 16, 32, 64, 128]):
        self.stacked_convs = stacked_convs
        super(RetinaHead, self).__init__()
        self.anchor_bases = [generate_anchor_base(base_size,
                                                  ratios=ratios,
                                                  scales=None,
                                                  octave_base_scale=octave_base_scale,
                                                  scales_per_octave=scales_per_octave
                                                  ) 
                                                  for base_size in feat_strides]
        n_anchor = self.anchor_bases[0].shape[0]

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else feat_channels
            self.cls_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))
            self.reg_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))

        self.retina_cls = nn.Conv(feat_channels,n_anchor * num_classes,3,padding=1)
        self.retina_reg = nn.Conv(feat_channels, n_anchor * 4, 3, padding=1)

    def execute_single(self, x,anchor_base,feat_stride):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        n, _, hh, ww = x.shape
        
        anchor = grid_anchors(anchor_base,feat_stride, (hh, ww))
        anchor = jt.array(anchor)
        
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred

    def loss(self,rpn_loc, rpn_score, anchor, targets):
        gt_rpn_locs = []
        gt_rpn_labels = []
        for target in targets:
            gt_bbox = target["bboxes"]
            img_size = target["img_size"]
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(gt_bbox,anchor,img_size)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            
        gt_rpn_labels = jt.contrib.concat(gt_rpn_labels,dim=0)
        gt_rpn_locs = jt.contrib.concat(gt_rpn_locs,dim=0)
        
        rpn_loc = rpn_loc.reshape(-1,4)
        rpn_score  = rpn_score.reshape(-1,2)
        
        rpn_loc_loss = faster_rcnn_loss(rpn_loc,gt_rpn_locs,gt_rpn_labels,beta=self.rpn_beta)
        rpn_cls_loss = sigmoid_focal_loss(rpn_score[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
        return rpn_loc_loss,rpn_cls_loss
    
    def execute(self,xs,targets):
        img_size = [t["img_size"] for t in targets]
        proposals = []
        rpn_locs = []
        rpn_scores = []
        anchors = []
        
        for x,anchor_base,feat_stride in zip(xs,self.anchor_bases,self.feat_strides):
            rpn_loc, rpn_score, proposal, anchor = self.execute_single(x,anchor_base,feat_stride,img_size)
            rpn_locs.append(rpn_loc)
            rpn_scores.append(rpn_score)
            anchors.append(anchor)
            proposals.append(proposal)
        
        rpn_locs = jt.contrib.concat(rpn_locs,dim=1)
        rpn_scores = jt.contrib.concat(rpn_scores,dim=1)
        anchors = jt.contrib.concat(anchors,dim=0)
        
        losses = dict()
        if self.is_training():
            rpn_loc_loss,rpn_cls_loss = self.loss(rpn_locs, rpn_scores, anchors,targets)
            losses = dict(
                rpn_loc_loss = rpn_loc_loss,
                rpn_cls_loss = rpn_cls_loss
            )
        
        return proposals,losses
