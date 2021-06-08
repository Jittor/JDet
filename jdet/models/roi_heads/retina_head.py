import math

from jdet.models.losses.focal_loss import sigmoid_focal_loss
from jdet.models.losses.smooth_l1_loss import smooth_l1_loss
from jdet.models.roi_heads.anchor_generator import bbox2loc, bbox_iou, generate_anchor_base, grid_anchors, loc2bbox
from jdet.utils.registry import ROI_HEADS
from jittor import nn,init 
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
                 n_class,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 ratios=[0.5, 1.0, 2.0],
                 feat_strides=[8, 16, 32, 64, 128],
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.4,
                 neg_iou_thresh_lo=0.,
                 nms_pre = 1000,
                 max_dets = 100,
                 ):
        super(RetinaHead, self).__init__()

        self.feat_strides = feat_strides
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.stacked_convs = stacked_convs
        self.nms_pre = nms_pre
        self.max_dets = max_dets

        
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
            chn = in_channels if i == 0 else feat_channels
            self.cls_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))
            self.reg_convs.append(nn.Conv(chn,feat_channels,3,stride=1,padding=1))
        
        self.n_class = n_class
        self.retina_cls = nn.Conv(feat_channels,n_anchor * n_class,3,padding=1)
        self.retina_reg = nn.Conv(feat_channels, n_anchor * 4, 3, padding=1)

        self.roi_beta = 0.
        self.nms_thresh = 0.5
        self.score_thresh = 0.05

        self.init_weights()

    def init_weights(self):
        
        # Initialization
        for modules in [self.cls_convs, self.reg_convs]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv):
                    init.gauss_(layer.weight, mean=0, std=0.01)
                    init.constant_(layer.bias, 0)

        
        init.constant_(self.retina_reg.bias, 0)
        init.gauss_(self.retina_reg.weight,0,0.01)
        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - 0.01) / 0.01))
        init.constant_(self.retina_cls.bias, bias_value)
        init.gauss_(self.retina_cls.weight,0,0.01)

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
            cls_feat = nn.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = nn.relu(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        bbox_pred = bbox_pred.permute(0,2,3,1).reshape(-1,4)
        cls_score = cls_score.permute(0,2,3,1).reshape(-1,self.n_class)

        return bbox_pred, cls_score, anchor
    
    def assign_labels(self,roi, bbox, label):
        iou = bbox_iou(roi, bbox)
        gt_assignment,max_iou = iou.argmax(dim=1)

        gt_roi_label = -jt.ones((gt_assignment.shape[0],))

        pos_index = jt.where(max_iou >= self.pos_iou_thresh)[0]

        neg_index = jt.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]        
        
        gt_roi_label[neg_index] = 0  # negative labels --> 0
        gt_roi_label[pos_index] = label[gt_assignment[pos_index]]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = jt.zeros(roi.shape)
        gt_roi_loc[pos_index] = bbox2loc(roi[pos_index], bbox[gt_assignment[pos_index]])

        return gt_roi_loc, gt_roi_label

    def get_bboxes(self,indexes,proposals,bbox_pred,score,targets):
        cls_bbox = loc2bbox(proposals,bbox_pred)
        probs = score.sigmoid()
        results = []

        for i,target in enumerate(targets):
            img_size = target["img_size"]
            ori_img_size = target["ori_img_size"]

            index = jt.where(indexes==i)[0]
            score = probs[index,:]
            bbox = cls_bbox[index,:]
            bbox[:,0::2] = jt.clamp(bbox[:,0::2],min_v=0,max_v=img_size[0])*(ori_img_size[0]/img_size[0])
            bbox[:,1::2] = jt.clamp(bbox[:,1::2],min_v=0,max_v=img_size[1])*(ori_img_size[1]/img_size[1])
            boxes = []
            scores = []
            labels = []
            for j in range(self.n_class):
                score_j = score[:,j]
                bbox_j = bbox
                mask = jt.where(score_j>self.score_thresh)[0]
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]

                if score_j.numel()>self.nms_pre:
                    order,_ = jt.argsort(score_j, descending=True)
                    order = order[:self.nms_pre]
                    score_j = score_j[order]
                    bbox_j = bbox_j[order]

                dets = jt.contrib.concat([bbox_j,score_j.unsqueeze(1)],dim=1)
                keep = jt.nms(dets,self.nms_thresh)
                bbox_j = bbox_j[keep]
                score_j = score_j[keep]
                label_j = jt.ones_like(score_j).int32()*(j+1)
                boxes.append(bbox_j)
                scores.append(score_j)
                labels.append(label_j)
            
            boxes = jt.contrib.concat(boxes,dim=0)
            scores = jt.contrib.concat(scores,dim=0)
            labels = jt.contrib.concat(labels,dim=0)
            if scores.numel()>self.max_dets:
                order,_ = jt.argsort(scores,descending=True)
                order = order[:self.max_dets]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]
            
            results.append(dict(
                boxes=boxes,
                scores=scores,
                labels=labels,
                img_id=target["img_id"]))
        
        return results

    def losses(self,all_bbox_pred,all_cls_score,all_gt_roi_locs,all_gt_roi_labels):
        normalizer = max((all_gt_roi_labels>0).sum().item(),1)
        
        # only calculate the positive box,if beta==0. means L1 loss
        roi_loc_loss = smooth_l1_loss(all_bbox_pred[all_gt_roi_labels>0],all_gt_roi_locs[all_gt_roi_labels>0],beta=self.roi_beta,reduction="sum")

        # build one hot with background
        inputs = all_cls_score[all_gt_roi_labels>=0]
        cates = all_gt_roi_labels[all_gt_roi_labels>=0].unsqueeze(1)
        class_range = jt.index((self.n_class,),dim=0)+1
        cates = (cates==class_range)
        roi_cls_loss = sigmoid_focal_loss(inputs,cates,reduction="sum")

        losses = dict(
            roi_cls_loss=roi_cls_loss/normalizer,
            roi_loc_loss=roi_loc_loss/normalizer
        ) 

        return losses

    def execute(self,xs,targets):
        all_bbox_pred = []
        all_cls_score = []
        all_proposals = []
        all_gt_roi_labels = []
        all_gt_roi_locs = []
        all_indexes = []
        for x,anchor_base,feat_stride in zip(xs,self.anchor_bases,self.feat_strides):
            bbox_pred, cls_score, anchor = self.execute_single(x,anchor_base,feat_stride)
            gt_roi_locs = []
            gt_roi_labels = []
            proposals = []
            indexes = []
            for i,target in enumerate(targets):
                if self.is_training():
                    gt_bbox = target["bboxes"]
                    gt_label = target["labels"]
                    gt_roi_loc,gt_roi_label= self.assign_labels(anchor,gt_bbox,gt_label)
                    gt_roi_locs.append(gt_roi_loc)
                    gt_roi_labels.append(gt_roi_label)

                index = i*jt.ones((anchor.shape[0],))
                indexes.append(index)
                proposals.append(anchor)


            proposals = jt.contrib.concat(proposals,dim=0)
            indexes = jt.contrib.concat(indexes,dim=0)

            all_bbox_pred.append(bbox_pred)
            all_cls_score.append(cls_score)
            all_proposals.append(proposals)
            all_indexes.append(indexes)

            if self.is_training():
                gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
                gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)
                all_gt_roi_locs.append(gt_roi_locs)
                all_gt_roi_labels.append(gt_roi_labels)
        

        all_bbox_pred = jt.contrib.concat(all_bbox_pred,dim=0)
        all_cls_score = jt.contrib.concat(all_cls_score,dim=0)
        all_proposals = jt.contrib.concat(all_proposals,dim=0)
        all_indexes = jt.contrib.concat(all_indexes,dim=0)
    
        if self.is_training():
            all_gt_roi_locs = jt.contrib.concat(all_gt_roi_locs,dim=0)
            all_gt_roi_labels = jt.contrib.concat(all_gt_roi_labels,dim=0)
        
        losses = dict()
        results = []
        if self.is_training():
            losses = self.losses(all_bbox_pred,all_cls_score,all_gt_roi_locs,all_gt_roi_labels)
        else:
            results = self.get_bboxes(all_indexes,all_proposals,all_bbox_pred,all_cls_score,targets)

        return results,losses 
        