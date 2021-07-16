from http.client import UnimplementedFileMode
from jittor import nn
from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, SHARED_HEADS, ROI_EXTRACTORS
from jdet.ops.bbox_transfomrs import bbox2roi


@MODELS.register_module()
class FasterRCNN(nn.Module):

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
            losses (dict): losses
        '''
        losses = dict()
        features = self.backbone(images)
        if(self.neck):
            features = self.neck(features)
        proposal_list, rpn_loss = self.rpn(features, targets)
        losses.update(rpn_loss)

        bbox_assigner = build_from_cfg(self.train_cfg.rcnn.assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn.sampler, BOXES) #ingnored: context=self
        sampling_results = []
        for proposal, target in zip(proposal_list, targets):
            assign_result = bbox_assigner.assign(
                proposal, target['bboxes'], target['bboxes_ignore'], target['labels']
            )
            sampling_result = bbox_sampler.sample(
                assign_result, proposal, target['bboxes'], target['labels']
                #ignored: feats=[lvl_feat[i][None] for lvl_feat in images])
            )
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            images[:self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_targets = self.bbox_head.get_target(
            sampling_results, self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                        *bbox_targets)

        losses.update(loss_bbox)
        return losses
    
    def excute(self, images, targets=None):
        if self.is_training():
            return self.excute_train(images, targets)
        else:
            #TODO
            raise NotImplementedError
