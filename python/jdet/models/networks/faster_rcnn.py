from http.client import UnimplementedFileMode
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, SHARED_HEADS, ROI_EXTRACTORS

@MODELS.register_module
class FasterRCNN(nn):

    def __init__(self,
                 backbone,
                 rpn,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS)
        self.shared_head = build_from_cfg(shared_head, SHARED_HEADS)
        self.rpn = build_from_cfg(rpn, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def excute_train(self, images, targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections
            losses (dict): losses
        '''
        losses = dict()
        features = self.backbone(images)
        if(self.neck):
            features = self.neck(features)
        proposal_list, rpn_loss = self.rpn(features, targets)
        losses.update(rpn_loss)

        #TODO: sampling and bbox_roi_extractor
        sampling_results = self.sampling(proposal_list, features, targets)
        loss_bbox = self.bbox_roi_extractor(sampling_results, features, targets)

        losses.update(loss_bbox)
        return losses
    
    def excute(self, images, targets=None):
        if self.is_training():
            return self.excute_train(images, targets)
        else:
            #TODO
            raise NotImplementedError
