import jittor as jt 
from jittor import nn,init 
from jdet.utils.registry import ROI_HEADS
from jdet.ops.roi_align import ROIAlign
import numpy as np
from jdet.models.losses.faster_rcnn_loss import faster_rcnn_loss
from .anchor_generator import ProposalTargetCreator, loc2bbox


@ROI_HEADS.register_module()
class BoxHead(nn.Module):
    """
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
    """

    def __init__(self,  in_channels=256,
                        n_class=81, 
                        roi_size=7, 
                        feat_strides = [16],
                        proposal_target_cfg=dict(
                            n_sample=128,
                            pos_ratio=0.25, 
                            pos_iou_thresh=0.5,
                            neg_iou_thresh_hi=0.5, 
                            neg_iou_thresh_lo=0.0)
                        ):
        # n_class includes the background
        super(BoxHead, self).__init__()

        self.proposal_target_creator = ProposalTargetCreator(**proposal_target_cfg)
        
        self.fc = nn.Sequential(
                nn.Linear(in_channels * roi_size * roi_size, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.fc_channels = in_channels * roi_size * roi_size

        self.n_class = n_class
        self.roi_aligns = [ROIAlign(roi_size,1./f_s) for f_s in feat_strides]
        
        self.roi_beta = 1.
        self.nms_thresh = 0.5
        self.score_thresh = 0.05

        self.init_weights()
    
    def init_weights(self):
        init.gauss_(self.cls_loc.weight,0,0.001)
        init.constant_(self.cls_loc.bias,0)
        init.gauss_(self.score.weight,0,0.01)
        init.constant_(self.score.bias,0)


    def execute(self,xs,all_proposals,targets):
        # use features for roi_align nums
        xs = xs[:len(self.roi_aligns)]
        all_proposals = all_proposals[:len(self.roi_aligns)]

        if self.is_training():
            all_level_proposals = []
            all_level_indexes = []
            all_level_gt_locs = []
            all_level_gt_labels = []
            for one_level_proposals in all_proposals:
                gt_roi_locs = []
                gt_roi_labels = []
                proposals = []
                indexes = []
                for i,(proposal,target) in enumerate(zip(one_level_proposals,targets)):
                    gt_bbox = target["bboxes"]
                    gt_label = target["labels"]
                    proposal,gt_roi_loc,gt_roi_label= self.proposal_target_creator(proposal,gt_bbox,gt_label)
                    index = i*jt.ones((proposal.shape[0],1))
                    indexes.append(index)
                    proposals.append(proposal)
                    gt_roi_locs.append(gt_roi_loc)
                    gt_roi_labels.append(gt_roi_label)

                gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
                gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)

                all_level_proposals.append(proposals)
                all_level_indexes.append(indexes)
                all_level_gt_labels.append(gt_roi_labels)
                all_level_gt_locs.append(gt_roi_locs)
            
        else:
            all_level_proposals = all_proposals

        features = []
        all_rois = []
        for roi_align,x,proposals in zip(self.roi_aligns,xs,all_level_proposals):
            rois = []
            indexes = []
            for i,proposal in enumerate(proposals):
                index = i*jt.ones((proposal.shape[0],1))
                indexes.append(index)
                rois.append(proposal)
            indexes = jt.contrib.concat(indexes,dim=0)
            rois = jt.contrib.concat(rois,dim=0)
            index_rois = jt.contrib.concat([indexes,rois],dim=1)
            all_rois.append(index_rois)

            level_feature = roi_align(x, index_rois)
            features.append(level_feature)
        
        features = jt.contrib.concat(features,dim=0)

        features = features.reshape(features.shape[0],self.fc_channels)
        fc7 = self.fc(features)
        roi_cls_loc = self.cls_loc(fc7)
        roi_score = self.score(fc7)

        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
        
        losses = dict()
        results = []
        if self.is_training():

            all_level_gt_labels = jt.contrib.concat(all_level_gt_labels,dim=0)
            all_level_gt_locs = jt.contrib.concat(all_level_gt_locs,dim=0)
        
            roi_loc = roi_cls_loc[jt.index((all_level_gt_labels.shape[0],),dim=0), all_level_gt_labels]
            roi_loc_loss = faster_rcnn_loss(roi_loc,all_level_gt_locs,all_level_gt_labels,beta=self.roi_beta)
            roi_cls_loss = nn.cross_entropy_loss(roi_score, all_level_gt_labels)

            losses = dict(
                roi_cls_loss=roi_cls_loss,
                roi_loc_loss=roi_loc_loss
            )

        else:
            all_rois = jt.contrib.concat(all_rois,dim=0)
            indexes = all_rois[:,0]
            rois = all_rois[:,1:]
            probs = nn.softmax(roi_score,dim=-1)
            rois = rois.unsqueeze(1).repeat(1,self.n_class,1)
            cls_bbox = loc2bbox(rois.reshape(-1,4),roi_cls_loc.reshape(-1,4))
            cls_bbox = cls_bbox.reshape(-1,self.n_class,4)
            
            for i,target in enumerate(targets):
                img_size = target["img_size"]
                ori_img_size = target["ori_img_size"]

                index = jt.where(indexes==i)[0]
                score = probs[index,:]
                bbox = cls_bbox[index,:,:]
                bbox[:,:,0::2] = jt.clamp(bbox[:,:,0::2],min_v=0,max_v=img_size[0])*(ori_img_size[0]/img_size[0])
                bbox[:,:,1::2] = jt.clamp(bbox[:,:,1::2],min_v=0,max_v=img_size[1])*(ori_img_size[1]/img_size[1])
                boxes = []
                scores = []
                labels = []
                for j in range(1,self.n_class):
                    bbox_j = bbox[:,j,:]
                    score_j = score[:,j]
                    mask = jt.where(score_j>self.score_thresh)[0]
                    bbox_j = bbox_j[mask,:]
                    score_j = score_j[mask]
                    dets = jt.contrib.concat([bbox_j,score_j.unsqueeze(1)],dim=1)
                    keep = jt.nms(dets,self.nms_thresh)
                    bbox_j = bbox_j[keep]
                    score_j = score_j[keep]
                    label_j = jt.ones_like(score_j).int32()*j
                    boxes.append(bbox_j)
                    scores.append(score_j)
                    labels.append(label_j)
                
                boxes = jt.contrib.concat(boxes,dim=0)
                scores = jt.contrib.concat(scores,dim=0)
                labels = jt.contrib.concat(labels,dim=0)
                results.append(dict(
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    img_id=target["img_id"]))

        return results,losses 
        