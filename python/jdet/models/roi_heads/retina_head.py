import math

from jdet.models.losses.focal_loss import sigmoid_focal_loss
from jdet.models.losses.smooth_l1_loss import smooth_l1_loss
from jdet.models.boxes.box_ops import bbox2loc, bbox_iou, loc2bbox, loc2bbox_r, bbox2loc_r
from jdet.ops import box_iou_rotated
from jdet.utils.registry import HEADS
from jittor import nn,init 
import jittor as jt
from jdet.utils.registry import build_from_cfg,BOXES
import numpy as np
import jdet
from jdet.models.boxes.box_ops import rotated_box_to_bbox, boxes_xywh_to_x0y0x1y1, boxes_x0y0x1y1_to_xywh, rotated_box_to_poly
# from  import get_var

@HEADS.register_module()
class RetinaHead(nn.Module):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> self = RetinaHead(11, 7)
        >>> x = jt.randn(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
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
                 mode='H',
                 score_threshold = 0.05,
                 nms_iou_threshold = 0.5,
                 roi_beta = 0.,
                 cls_loss_weight=1.,
                 loc_loss_weight=0.2,
                 ):
        super(RetinaHead, self).__init__()

        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo
        self.stacked_convs = stacked_convs
        self.nms_pre = nms_pre
        self.max_dets = max_dets
        self.mode = mode
        
        self.anchor_generator = build_from_cfg(anchor_generator, BOXES)
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

    def execute_single(self, x):
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
        
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = nn.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = nn.relu(reg_conv(reg_feat))
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        bbox_pred = bbox_pred.permute(0,2,3,1).reshape(n,-1,5)
        cls_score = cls_score.permute(0,2,3,1).reshape(n,-1,self.n_class)

        return bbox_pred, cls_score
    
    def assign_labels(self,roi, bbox, bbox_h, label):
        #roi:x0y0x1y1a
        #bbox:xywha
        if (self.mode == 'H'):
            iou = bbox_iou(roi, bbox)
        else:
            if (self.anchor_mode == 'H'):
                iou = bbox_iou(roi[:, :4], bbox_h)
            else:
                iou = box_iou_rotated(roi, bbox) #TODO check
        gt_assignment,max_iou = iou.argmax(dim=1)

        gt_roi_label = -jt.ones((gt_assignment.shape[0],))

        pos_index = jt.where(max_iou >= self.pos_iou_thresh)[0]

        neg_index = jt.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]        
        
        gt_roi_label[neg_index] = 0  # negative labels --> 0
        gt_roi_label[pos_index] = label[gt_assignment[pos_index]]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = jt.zeros(roi.shape)
        if (self.mode == 'H'):
            gt_roi_loc[pos_index] = bbox2loc(roi[pos_index], bbox[gt_assignment[pos_index]])
        else:
            bbox_ = bbox
            roi_ = self.cvt2_w_greater_than_h(boxes_x0y0x1y1_to_xywh(roi))
            gt_roi_loc[pos_index] = bbox2loc_r(roi_[pos_index], bbox_[gt_assignment[pos_index]])

        return gt_roi_loc, gt_roi_label

    # input xywha(pi)  [-pi/2,0)
    # output xywha(pi) [-pi,0)
    def cvt2_w_greater_than_h(self, boxes, reverse_hw=True):
        boxes = boxes.copy()
        if (reverse_hw): #TODO: yxe?
            x, y, w, h, a = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            boxes = jt.stack([x, y, h, w, a], dim=1)

        h = boxes[:, 3:4]
        w = boxes[:, 2:3]
        
        remain_mask = w > h
        convert_mask = 1 - remain_mask
        remain_coords = boxes * remain_mask.reshape([-1, 1])

        boxes[:, [2, 3]] = boxes[:, [3, 2]]
        boxes[:, 4] += 0.5 * np.pi

        convert_coords = boxes * convert_mask.reshape([-1, 1])
        boxes = remain_coords + convert_coords

        boxes[:, 4] -= 0.5 * np.pi
        return boxes

    def get_bboxes(self,proposals_,bbox_pred_,score_,targets):
        results = []
        for i,target in enumerate(targets):
            if (self.mode == "H"):
                cls_bbox = loc2bbox(proposals_[i],bbox_pred_[i])
            else:
                proposals = boxes_x0y0x1y1_to_xywh(proposals_[i])
                proposals = self.cvt2_w_greater_than_h(proposals)
                proposals[:, 4] += 0.5 * np.pi#TODO: yxe?
                cls_bbox = loc2bbox_r(proposals,bbox_pred_[i])

            probs = score_[i].sigmoid()

            img_size = target["img_size"]
            ori_img_size = target["ori_img_size"]
            assert(abs(ori_img_size[0]/ori_img_size[1] - img_size[0]/img_size[1]) < 1e-6) # too keep the angle

            score = probs
            bbox = cls_bbox
            # bbox[:,[0,2]] = jt.clamp(bbox[:,[0,2]],min_v=0,max_v=img_size[0])*(ori_img_size[0]/img_size[0])
            # bbox[:,[1,3]] = jt.clamp(bbox[:,[1,3]],min_v=0,max_v=img_size[1])*(ori_img_size[1]/img_size[1])
            
            bbox[:,[0,2]] = bbox[:,[0,2]]*(ori_img_size[0]/img_size[0])
            bbox[:,[1,3]] = bbox[:,[1,3]]*(ori_img_size[1]/img_size[1])
            
            boxes = []
            scores = []
            labels = []
            for j in range(self.n_class):
                score_j = score[:,j]
                bbox_j = bbox
                # #TODO Err
                # mask = ((score_j>self.score_thresh)+(bbox_j[:, 4] < 90) + (bbox_j[:, 4] > -90))==3
                # bbox_j = bbox_j[mask,:]
                # score_j = score_j[mask]
                #Correct
                mask = score_j>self.score_thresh
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]
                jt.sync([bbox_j, score_j])
                mask = bbox_j[:, 4] < 0.5 * np.pi
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]
                jt.sync([bbox_j, score_j])
                mask = bbox_j[:, 4] > -0.5 * np.pi
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]
                jt.sync([bbox_j, score_j])

                if score_j.numel()>self.nms_pre: #TODO check nms_pre
                    order,_ = jt.argsort(score_j, descending=True)
                    order = order[:self.nms_pre]
                    score_j = score_j[order]
                    bbox_j = bbox_j[order]

                dets = jt.contrib.concat([bbox_j,score_j.unsqueeze(1)],dim=1)
                if (self.mode == 'H'):
                    keep = jt.nms(dets,self.nms_thresh) #TODO check
                else:
                    keep = jdet.ops.nms_rotated.nms_rotated(bbox_j,score_j,self.nms_thresh)
                bbox_j = bbox_j[keep] #cx,cy,w,h,a(pi)
                score_j = score_j[keep]
                label_j = jt.ones_like(score_j).int32()*j
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
            polys = rotated_box_to_poly(boxes)
            results.append((polys, scores, labels))
        return results

    def losses(self,all_bbox_pred_,all_cls_score_,all_gt_roi_locs_,all_gt_roi_labels_):
        batch_size = len(all_bbox_pred_)
        losses = dict(
            roi_cls_loss=0,
            roi_loc_loss=0
        ) 
        for i in range(batch_size):
            all_gt_roi_labels = all_gt_roi_labels_[i]
            normalizer = max((all_gt_roi_labels>0).sum().item(),1)

            # only calculate the positive box,if beta==0. means L1 loss
            roi_loc_loss = smooth_l1_loss(all_bbox_pred_[i][all_gt_roi_labels>0],all_gt_roi_locs_[i][all_gt_roi_labels>0],beta=self.roi_beta,reduction="sum")

            # build one hot with background
            inputs = all_cls_score_[i][all_gt_roi_labels>=0]
            cates = all_gt_roi_labels[all_gt_roi_labels>=0]#.unsqueeze(1)

            roi_cls_loss = sigmoid_focal_loss(inputs,cates,reduction="sum",alpha=0.25)
            losses['roi_cls_loss'] += roi_cls_loss/normalizer
            losses['roi_loc_loss'] += roi_loc_loss/normalizer
        losses['roi_cls_loss'] *= self.cls_loss_weight / batch_size
        losses['roi_loc_loss'] *= self.loc_loss_weight / batch_size
        return losses

    def execute(self,xs,targets):
        n = len(targets)
        all_bbox_pred = []
        all_cls_score = []
        all_proposals = []
        all_gt_roi_labels = []
        all_gt_roi_locs = []
        for i in range(n):
            all_bbox_pred.append([])
            all_cls_score.append([])
            all_proposals.append([])
            all_gt_roi_labels.append([])
            all_gt_roi_locs.append([])


        sizes = []
        for x in xs:
            sizes.append([x.shape[2], x.shape[3]])
        anchors = self.anchor_generator.grid_anchors(sizes)

        id = -1
        for x in xs:
            id += 1
            bbox_pred, cls_score = self.execute_single(x)
            anchor = anchors[id] #x0,y0,x1,y1

            anchor = jt.contrib.concat([anchor, -0.5 * np.pi * jt.ones([anchor.shape[0], 1])], 1)

            for i,target in enumerate(targets):
                if self.is_training():
                    gt_bbox = target["rboxes"]#xywha
                    gt_bbox_h = target["rboxes_h"]#xywha
                    gt_label = target["labels"]

                    gt_roi_loc,gt_roi_label= self.assign_labels(anchor,gt_bbox,gt_bbox_h,gt_label)
                    all_gt_roi_locs[i].append(gt_roi_loc)
                    all_gt_roi_labels[i].append(gt_roi_label)

                all_proposals[i].append(anchor)
                all_bbox_pred[i].append(bbox_pred[i])
                all_cls_score[i].append(cls_score[i])

        for i in range(n):
            all_bbox_pred[i] = jt.concat(all_bbox_pred[i], 0)
            all_cls_score[i] = jt.concat(all_cls_score[i], 0)
            all_proposals[i] = jt.concat(all_proposals[i], 0)
            if self.is_training():
                all_gt_roi_locs[i] = jt.concat(all_gt_roi_locs[i],0)
                all_gt_roi_labels[i] = jt.concat(all_gt_roi_labels[i],0)

        losses = dict()
        results = []
        if self.is_training():
            losses = self.losses(all_bbox_pred,all_cls_score,all_gt_roi_locs,all_gt_roi_labels)
        else:
            results = self.get_bboxes(all_proposals,all_bbox_pred,all_cls_score,targets)

        return results,losses 
        