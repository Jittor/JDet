from http.client import UnimplementedFileMode
from jittor import nn
from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, SHARED_HEADS, ROI_EXTRACTORS
from jdet.ops.bbox_transfomrs import bbox2roi


@MODELS.register_module()
class FasterRCNN(nn.Module):

    def __init__(self,
                 backbone,
                 rpn_head,
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
        self.rpn_head = build_from_cfg(rpn_head, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    

    def excute_train(self,
                     images,
                     image_meta,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None):
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

        rpn_outs = self.rpn_head(images)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta,
                                        self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (image_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        bbox_assigner = build_from_cfg(self.train_cfg.rcnn.assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn.sampler, BOXES) #ingnored: context=self
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for proposal, gt_bbox, gt_bbox_ignore, gt_label in zip(proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels):
            assign_result = bbox_assigner.assign(
                proposal, gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = bbox_sampler.sample(
                assign_result, proposal, gt_bbox, gt_label
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
