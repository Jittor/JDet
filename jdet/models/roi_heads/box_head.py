import jittor as jt 
from jittor import nn,init 
from jdet.utils.registry import ROI_HEADS
from jdet.ops.roi_align import ROIAlign
import numpy as np
from jdet.models.losses.faster_rcnn_loss import _fast_rcnn_loc_loss
from .anchor_generator import loc2bbox

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

    def __init__(self, in_channels=256,n_class=81, roi_size=7, spatial_scales=[1./4, 1./8, 1./16, 1./32, 1./64]):
        # n_class includes the background
        super(BoxHead, self).__init__()

        self.classifier = nn.Sequential(
                nn.Linear(in_channels * roi_size * roi_size, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.n_class = n_class
        self.roi_aligns = [ROIAlign(roi_size,spatial_scale) for spatial_scale in self.spatial_scales]
        
        self.roi_sigma = 1.

        self.init_weights()
    
    def init_weights(self):
        init.gauss_(self.cls_loc.weight,0,0.001)
        init.constant_(self.cls_loc.bias,0)
        init.gauss_(self.score.weight,0,0.01)
        init.constant_(self.score.bias,0)
            

    def execute_single(self, x, rois, roi_indices,roi_align):
        """Forward the chain.
        We assume that there are :math:`N` batches.
        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.
        """
        indices_and_rois = jt.contrib.concat([roi_indices.unsqueeze(1), rois], dim=1)
        pool = roi_align(x, indices_and_rois)
        pool = pool.view(pool.shape[0], np.prod(pool.shape[1:]).item())
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

    def loss_single(self,roi_cls_loc, roi_score,gt_roi_locs,gt_roi_labels):
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, np.prod(roi_cls_loc.shape[1:]).item()//4, 4)
        roi_loc = roi_cls_loc[:, gt_roi_labels]
        
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc,gt_roi_locs,gt_roi_labels,self.roi_sigma)
        roi_cls_loss = nn.cross_entropy_loss(roi_score, gt_roi_labels) 

        losses = {
            "roi_loc_loss":roi_loc_loss,
            "roi_cls_loss":roi_cls_loss
        }
        return losses

    def execute(self,xs,proposals,targets):
        losses = []
        outs = []
        for x,(sample_rois,sample_roi_indexs,gt_roi_locs,gt_roi_labels),roi_align in zip(xs,proposals,self.roi_aligns):
            roi_cls_locs, roi_scores = self.execute_single(x,sample_rois,sample_roi_indexs,roi_align)
            loss = self.loss_single(roi_cls_locs, roi_scores,gt_roi_locs,gt_roi_labels)
            outs.append((roi_cls_locs,roi_scores))
            losses.append(loss)
        return outs,losses
            
        