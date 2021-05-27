import jittor as jt 
from jittor import nn 
from jdet.utils.registry import ROI_HEADS

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

    def __init__(self, in_channels,n_class, roi_size, spatial_scale,sampling_ratio):
        # n_class includes the background
        super(RoIHead, self).__init__()

        self.classifier = nn.Sequential(
                nn.Linear(in_channels * roi_size * roi_size, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU()
            )
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = ROIAlign((self.roi_size, self.roi_size),self.spatial_scale,sampling_ratio=sampling_ratio)
        
        init.gauss_(self.cls_loc.weight,0,0.001)
        init.constant_(self.cls_loc.bias,0)
        init.gauss_(self.score.weight,0,0.01)
        init.constant_(self.score.bias,0)
            

    def execute(self, x, rois, roi_indices):
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
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.shape[0], np.prod(pool.shape[1:]).item())
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores