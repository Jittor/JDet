import jittor as jt 
from jittor import nn,init 
from jdet.utils.registry import ROI_HEADS
from jdet.ops.roi_align import ROIAlign
import numpy as np
from jdet.models.losses.faster_rcnn_loss import faster_rcnn_loss
from .anchor_generator import loc2bbox,bbox_iou,bbox2loc

class ProposalTargetCreator(nn.Module):
    """Assign ground truth bounding boxes to given RoIs.
    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.
    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, 
                 pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, 
                 neg_iou_thresh_lo=0.0
                 ):
        super(ProposalTargetCreator,self).__init__()
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn
        

    def execute(self, roi, bbox, label):
        """Assigns ground truth to sampled proposals.
        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.
        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.
        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
        Returns:
            (array, array, array):
            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.
        """
        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment,max_iou = iou.argmax(dim=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment]

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = jt.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.shape[0]))
        if pos_index.shape[0] > 0:
            tmp_indexes = np.arange(0,pos_index.shape[0])
            np.random.shuffle(tmp_indexes)
            tmp_indexes = tmp_indexes[:pos_roi_per_this_image]
            pos_index = pos_index[tmp_indexes]

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = jt.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.shape[0]))
        if neg_index.shape[0] > 0:
            tmp_indexes = np.arange(0,neg_index.shape[0])
            np.random.shuffle(tmp_indexes)
            tmp_indexes = tmp_indexes[:neg_roi_per_this_image]
            neg_index = neg_index[tmp_indexes]
        

        # The indices that we're selecting (both positive and negative).
        keep_index = jt.contrib.concat((pos_index, neg_index),dim=0)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])

        return sample_roi, gt_roi_loc, gt_roi_label


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
        self.nms_thresh = 0.3
        self.score_thresh = 0.01

        self.init_weights()
    
    def init_weights(self):
        init.gauss_(self.cls_loc.weight,0,0.001)
        init.constant_(self.cls_loc.bias,0)
        init.gauss_(self.score.weight,0,0.01)
        init.constant_(self.score.bias,0)
            

    def execute_single(self, x, proposals,roi_align,targets):
        # print(proposals[0])
        if self.is_training():
            rois = []
            indexes = []
            gt_roi_locs = []
            gt_roi_labels = []
            for i,(proposal,target) in enumerate(zip(proposals,targets)):
                gt_bbox = target["bboxes"]
                gt_label = target["labels"]
                proposal,gt_roi_loc,gt_roi_label= self.proposal_target_creator(proposal,gt_bbox,gt_label)
                index = i*jt.ones((proposal.shape[0],1))
                
                indexes.append(index)
                rois.append(proposal)
                gt_roi_locs.append(gt_roi_loc)
                gt_roi_labels.append(gt_roi_label)
            
            indexes = jt.contrib.concat(indexes,dim=0)
            rois = jt.contrib.concat(rois,dim=0)
            gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
            gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)
        else:
            rois = []
            indexes = []
            for i,proposal in enumerate(proposals):
                index = i*jt.ones((proposal.shape[0],1))
                indexes.append(index)
                rois.append(proposal)
            indexes = jt.contrib.concat(indexes,dim=0)
            rois = jt.contrib.concat(rois,dim=0)
        
        index_rois = jt.contrib.concat([indexes,rois],dim=1)
        feat = roi_align(x, index_rois)
        feat = feat.view(feat.shape[0], self.fc_channels)
        fc7 = self.fc(feat)
        roi_cls_loc = self.cls_loc(fc7)
        roi_score = self.score(fc7)
    
        roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)

        

        if self.is_training():
            roi_loc = roi_cls_loc[jt.index((gt_roi_labels.shape[0],),dim=0), gt_roi_labels]
            roi_loc_loss = faster_rcnn_loss(roi_loc,gt_roi_locs,gt_roi_labels,beta=self.roi_beta)
            roi_cls_loss = nn.cross_entropy_loss(roi_score, gt_roi_labels)
            return roi_cls_loss,roi_loc_loss
        
        probs = nn.softmax(roi_score,dim=-1)
        # print(rois)
        rois = rois.unsqueeze(1).repeat(1,self.n_class,1)
        cls_bbox = loc2bbox(rois.reshape(-1,4),roi_cls_loc.reshape(-1,4))
        cls_bbox = cls_bbox.reshape(-1,self.n_class,4)
        
        results = []
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
                boxes=boxes.numpy(),
                scores=scores.numpy(),
                labels=labels.numpy(),
                img_id=target["img_id"]))
        return results

    def execute(self,xs,proposals,targets):
        cls_losses = []
        loc_losses = []
        outs = []
        for x,proposal,roi_align in zip(xs,proposals,self.roi_aligns):
            if self.is_training():
                roi_cls_loss,roi_loc_loss = self.execute_single(x,proposal,roi_align,targets)
                cls_losses.append(roi_cls_loss)
                loc_losses.append(roi_loc_loss)
            else:
                results = self.execute_single(x,proposal,roi_align,targets)
                outs.append(results)
        
        results = dict(
            roi_losses = dict(
                roi_cls_loss = sum(cls_losses)/max(1,len(cls_losses)),
                roi_loc_loss = sum(loc_losses)/max(1,len(loc_losses))
            ),
            outs=outs
        )
        return results
            
        