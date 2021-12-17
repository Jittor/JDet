from http.client import UnimplementedFileMode
from jittor import nn
from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, SHARED_HEADS, ROI_EXTRACTORS
from jdet.ops.bbox_transforms import bbox2roi, dbbox2result
import jittor as jt
from jdet.utils.general import parse_losses

@MODELS.register_module()
class FasterRCNNOBB(nn.Module):

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
        super(FasterRCNNOBB, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS)
        self.shared_head = build_from_cfg(shared_head, SHARED_HEADS)
        self.rpn_head = build_from_cfg(rpn_head, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def execute_train(self, images, targets=None):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            losses (dict): losses
        '''
        self.backbone.train()

        image_meta = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_obbs = []
        for target in targets:
            meta = dict(
                ori_shape = target['ori_img_size'],
                img_shape = target['img_size'],
                pad_shape = target['pad_shape'],
                img_file = target['img_file'],
                to_bgr = target['to_bgr'],
                scale_factor = target['scale_factor']
            )
            image_meta.append(meta)
            gt_bboxes.append(target['hboxes'])
            gt_labels.append(target['labels'])
            gt_bboxes_ignore.append(target['hboxes_ignore'])
            gt_obbs.append(target['rboxes'])

        losses = dict()
        features = self.backbone(images)
        if(self.neck):
            features = self.neck(features)
        rpn_outs = self.rpn_head(features)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta,
                                        self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
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
                proposal[:,:4], gt_bbox, gt_bbox_ignore, gt_label
            )
            sampling_result = bbox_sampler.sample(
                assign_result, proposal, gt_bbox, gt_label  
            )
            sampling_results.append(sampling_result)

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(
            features[:self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        rbbox_targets = self.bbox_head.get_target(
            sampling_results, gt_obbs, gt_labels, self.train_cfg.rcnn)
        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                        *rbbox_targets)
        losses.update(loss_bbox)
        return losses

    def execute_test(self, images, targets=None, rescale=False):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            losses (dict): losses
        '''
        img_meta = []
        img_shape = []
        scale_factor = []
        for target in targets:
            ori_img_size = target['ori_img_size']
            meta = dict(
                ori_shape = ori_img_size,
                img_shape = ori_img_size,
                pad_shape = ori_img_size,
                scale_factor = target['scale_factor'],
                img_file = target['img_file']
            )
            img_meta.append(meta)
            img_shape.append(target['img_size'])
            scale_factor.append(target['scale_factor'])

        x = self.backbone(images)
        if(self.neck):
            x = self.neck(x)

        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        rois = bbox2roi(proposal_list)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=self.test_cfg.rcnn)

        bbox_results = dbbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return [bbox_results]
    
    def execute(self, images, targets=None):
        if self.is_training():
            return self.execute_train(images, targets)
        else:
            return self.execute_test(images, targets)