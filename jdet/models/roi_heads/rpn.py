import jittor as jt 
from jittor import nn 

from jdet.utils.registry import ROI_HEADS


import jittor as jt 
from jittor import nn,init
import numpy as np

from .anchor_generator import AnchorTargetCreator, ProposalCreator, generate_anchor_base,grid_anchors
from jdet.models.losses.faster_rcnn_loss import faster_rcnn_loss


@ROI_HEADS.register_module()
class RPN(nn.Module):

    def __init__(self, 
                in_channels=512, 
                mid_channels=512,
                ratios=[0.5, 1, 2],
                scales=[8], 
                feat_strides=[4, 8, 16, 32, 64],
                proposal_creator_cfg=dict(
                            nms_thresh=0.7,
                            n_train_pre_nms=2000,
                            n_train_post_nms=1000,
                            n_test_pre_nms=1000,
                            n_test_post_nms=1000,
                            min_size=0),
                anchor_target_cfg=dict(
                            n_sample=256,
                            pos_iou_thresh=0.7, 
                            neg_iou_thresh=0.3,
                            pos_ratio=0.5
                ),

                 
    ):
        super(RPN, self).__init__()
        self.anchor_bases = [generate_anchor_base(base_size,ratios,scales) for base_size in feat_strides]
        self.feat_strides = feat_strides
        self.proposal_layer = ProposalCreator(**proposal_creator_cfg)
        self.anchor_target_creator = AnchorTargetCreator(**anchor_target_cfg)

        n_anchor = self.anchor_bases[0].shape[0]
        self.conv1 = nn.Conv(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv(mid_channels, n_anchor * 4, 1, 1, 0)
        self._normal_init()
        self.rpn_beta = 1/9.        
        
    def _normal_init(self):
        for var in [self.conv1,self.score,self.loc]:
            init.gauss_(var.weight,0,0.01)
            init.constant_(var.bias,0.0)

    def execute_single(self, x, anchor_base,feat_stride,img_sizes):
        """Forward Region Proposal Network.
        Here are notations.
        * :math:`N` is batch size.
        * :math:`C` channel size of the input.
        * :math:`H` and :math:`W` are height and witdh of the input feature.
        * :math:`A` is number of anchors assigned to each pixel.
        Args:
            x : The Features extracted from images.
                Its shape is :math:`(N, C, H, W)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
        Returns:
            This is a tuple of five following values.
            * **rpn_locs**: Predicted bounding box offsets and scales for \
                anchors. Its shape is :math:`(N, H W A, 4)`.
            * **rpn_scores**:  Predicted foreground scores for \
                anchors. Its shape is :math:`(N, H W A, 2)`.
            * **rois**: A bounding box array containing coordinates of \
                proposal boxes.  This is a concatenation of bounding box \
                arrays from multiple images in the batch. \
                Its shape is :math:`(R', 4)`. Given :math:`R_i` predicted \
                bounding boxes from the :math:`i` th image, \
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            * **roi_indices**: An array containing indices of images to \
                which RoIs correspond to. Its shape is :math:`(R',)`.
            * **anchor**: Coordinates of enumerated shifted anchors. \
                Its shape is :math:`(H W A, 4)`.
        """
        n, _, hh, ww = x.shape
        
        anchor = grid_anchors(anchor_base,feat_stride, (hh, ww))
        anchor = jt.array(anchor)

        feat = nn.relu(self.conv1(x))
        rpn_loc = self.loc(feat)
        rpn_score = self.score(feat)

        rpn_loc = rpn_loc.permute(0, 2, 3, 1).reshape(n, -1, 4)
        rpn_score = rpn_score.permute(0, 2, 3, 1)
        rpn_score = rpn_score.reshape(n, -1, 2)
        rpn_fg_score = nn.softmax(rpn_score, dim=2)[...,1]
        
        proposals = []
        for i in range(n):
            proposal = self.proposal_layer(rpn_loc[i],rpn_fg_score[i],anchor,img_sizes[i])
            proposals.append(proposal)
        return rpn_loc, rpn_score, proposals, anchor

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
        rpn_cls_loss = nn.cross_entropy_loss(rpn_score[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
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
            
    