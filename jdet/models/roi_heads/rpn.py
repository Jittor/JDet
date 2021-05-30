import jittor as jt 
from jittor import nn 

from jdet.utils.registry import ROI_HEADS


import jittor as jt 
from jittor import nn,init
import numpy as np

from .anchor_generator import *
from .anchor_generator import _unmap
from jdet.models.losses import smooth_l1_loss

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


class AnchorTargetCreator(nn.Module):
    """Assign the ground truth bounding boxes to anchors.
    
    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.
    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, 
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        super(AnchorTargetCreator,self).__init__()
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def execute(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.
        Types of input arrays and output arrays are same.
        Here are notations.
        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.
        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.
        Returns:
            (array, array):
            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.
        """

        img_W, img_H = img_size

        n_anchor = len(anchor)
        inside_index = jt.where(
                (anchor[:, 0] >= 0) &
                (anchor[:, 1] >= 0) &
                (anchor[:, 2] <= img_W) &
                (anchor[:, 3] <= img_H)
                )[0]
        if inside_index.sum().item()==0:
            return None,None

        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = -jt.ones((anchor.shape[0],), dtype="int32")

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1
        
        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = jt.where(label == 1)[0]
        if len(pos_index) > n_pos:
            tmp_index = np.arange(0,pos_index.shape[0])
            np.random.shuffle(tmp_index)
            disable_index = tmp_index[:pos_index.shape[0] - n_pos]
            disable_index = pos_index[disable_index]
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - jt.sum(label == 1).item()
        neg_index = jt.where(label == 0)[0]
        if len(neg_index) > n_neg:
            tmp_index = np.arange(0,neg_index.shape[0])
            np.random.shuffle(tmp_index)
            disable_index = tmp_index[:neg_index.shape[0] - n_neg]
            disable_index = neg_index[disable_index]
            label[disable_index] = -1
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious,max_ious = ious.argmax(dim=1)
        gt_argmax_ious,gt_max_ious = ious.argmax(dim=0)
        gt_argmax_ious = jt.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


    
class ProposalCreator(nn.Module):
    """Proposal regions are generated by calling this object.
    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.
    """

    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        super(ProposalCreator,self).__init__()
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def execute(self, loc, score,anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.
        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.
        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.
        Type of the output is same as the inputs.
        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.
        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.
        """
        # NOTE: when test, remember
        if self.is_training():
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        roi[:,0] = jt.clamp(roi[:,0],min_v=0,max_v=img_size[0])
        roi[:,2] = jt.clamp(roi[:,2],min_v=0,max_v=img_size[0])
        
        roi[:,1] = jt.clamp(roi[:,1],min_v=0,max_v=img_size[1])
        roi[:,3] = jt.clamp(roi[:,3],min_v=0,max_v=img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = jt.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order,_ = jt.argsort(score, descending=True)
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).
        
        dets = jt.contrib.concat([roi,score.unsqueeze(1)],dim=1)
        keep = jt.nms(dets,self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


@ROI_HEADS.register_module()
class RPN(nn.Module):

    def __init__(self, 
                in_channels=512, 
                mid_channels=512,
                ratios=[0.5, 1, 2],
                anchor_scales=[8], 
                feat_strides=[4, 8, 16, 32, 64],
                proposal_creator_cfg=dict(
                            nms_thresh=0.7,
                            n_train_pre_nms=2000,
                            n_train_post_nms=1000,
                            n_test_pre_nms=1000,
                            n_test_post_nms=300,
                            min_size=16),
                anchor_target_cfg=dict(
                            n_sample=256,
                            pos_iou_thresh=0.7, 
                            neg_iou_thresh=0.3,
                            pos_ratio=0.5
                ),
                proposal_target_cfg=dict(
                            n_sample=128,
                            pos_ratio=0.25, 
                            pos_iou_thresh=0.5,
                            neg_iou_thresh_hi=0.5, 
                            neg_iou_thresh_lo=0.0
                ),
                 
    ):
        super(RPN, self).__init__()
        self.anchor_bases = generate_multilevel_anchor_base(base_sizes=feat_strides,scales=anchor_scales, ratios=ratios)
        self.feat_strides = feat_strides
        self.proposal_layer = ProposalCreator(**proposal_creator_cfg)
        self.anchor_target_creator = AnchorTargetCreator(**anchor_target_cfg)
        self.proposal_target_creator = ProposalTargetCreator(**proposal_target_cfg)

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

        n_anchor = anchor.shape[0] // (hh * ww)
        h = nn.relu(self.conv1(x))
        
        rpn_locs = self.loc(h)
       
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1)
        rpn_softmax_scores = nn.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1]
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)
        rois = []
        roi_indices = []
        for i in range(n):
            roi = self.proposal_layer(
                                    rpn_locs[i],
                                    rpn_fg_scores[i],
                                    anchor, 
                                    img_sizes[i],
                                    scale=1.)
            batch_index = i * jt.ones((len(roi),), dtype='int32')
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = jt.contrib.concat(rois, dim=0)
        roi_indices = jt.contrib.concat(roi_indices, dim=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    def loss_single(self,rpn_locs, rpn_scores, rois, roi_indices, anchor,targets):
        sample_rois = []
        gt_roi_locs = []
        gt_roi_labels = []
        sample_roi_indexs = []
        gt_rpn_locs = []
        gt_rpn_labels = []
        for i,target in enumerate(targets):
            index = jt.where(roi_indices == i)[0]
            roi = rois[index,:]
            box = target["bboxes"]
            label = target["labels"]
            img_size = target["img_size"]
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi,box,label)
            sample_roi_index = i*jt.ones((sample_roi.shape[0],))
            
            sample_rois.append(sample_roi)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_locs.append(gt_roi_loc)
            sample_roi_indexs.append(sample_roi_index)
            
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(box,anchor,img_size)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            
        if any([label is None for label in gt_rpn_labels]):
            return None,None
        sample_roi_indexs = jt.contrib.concat(sample_roi_indexs,dim=0)
        sample_rois = jt.contrib.concat(sample_rois,dim=0)
        gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
        gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)
        
        # ------------------ RPN losses -------------------#
        rpn_locs = rpn_locs.reshape(-1,4)
        rpn_scores = rpn_scores.reshape(-1,2)
        gt_rpn_labels = jt.contrib.concat(gt_rpn_labels,dim=0)
        gt_rpn_locs = jt.contrib.concat(gt_rpn_locs,dim=0)
        
        weights = jt.zeros(gt_rpn_locs.shape)
        weight[gt_label>0,:]=1
        n_samples = (gt_label>=0).sum().float()
        rpn_loc_loss = smooth_l1_loss(rpn_locs,gt_rpn_locs,beta=self.rpn_beta,weight=weight,avg_factor=n_samples)

        rpn_cls_loss = nn.cross_entropy_loss(rpn_scores[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
        losses = {"rpn_loc_loss": rpn_loc_loss, 
                   "rpn_cls_loss": rpn_cls_loss}
        
        return losses,(sample_rois,sample_roi_indexs,gt_roi_locs,gt_roi_labels)

    def execute(self,xs,targets):
        img_size = [t["img_size"] for t in targets]
        losses = []
        outs = []
        for x,anchor_base,feat_stride in zip(xs,self.anchor_bases,self.feat_strides):
            rpn_locs, rpn_scores, rois, roi_indices, anchor  = self.execute_single(x,anchor_base,feat_stride,img_size)
            loss,out = self.loss_single(rpn_locs, rpn_scores, rois, roi_indices, anchor,targets)
            if loss is None:
                return None,None
            outs.append(out)
            losses.append(loss)
        return outs,losses
            

         
    