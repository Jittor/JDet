import jittor as jt
import copy
from jittor import nn
from jdet.utils.registry import BOXES, MODELS, build_from_cfg, BACKBONES, HEADS, NECKS, ROI_EXTRACTORS
from jdet.ops.bbox_transforms import bbox2roi, roi2droi, choose_best_Rroi_batch, dbbox2roi, dbbox2result

@MODELS.register_module()
class ReDet(nn.Module):
    def __init__(self,
                 backbone,
                 neck=None,
                #  shared_head=None,
                #  shared_head_rbbox=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                #  mask_roi_extractor=None,
                #  mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert backbone['type'] == 'ReResNet', 'ReDet only supports ReResNet backbone'
        assert neck['type'] == 'ReFPN', 'ReDet only supports ReFPN neck'
        super(ReDet, self).__init__()

        self.backbone = build_from_cfg(backbone, BACKBONES)
        self.neck = build_from_cfg(neck, NECKS)
        self.rpn_head = build_from_cfg(rpn_head, HEADS)
        # self.shared_head = build_from_cfg(shared_head, HEADS)
        # self.shared_head_rbbox = build_from_cfg(shared_head_rbbox, HEADS)
        self.bbox_roi_extractor = build_from_cfg(bbox_roi_extractor, ROI_EXTRACTORS)
        self.bbox_head = build_from_cfg(bbox_head, HEADS)
        self.rbbox_roi_extractor = build_from_cfg(rbbox_roi_extractor, ROI_EXTRACTORS)
        self.rbbox_head = build_from_cfg(rbbox_head, HEADS)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained

        self.init_weights()

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self):
        # super(ReDet, self).init_weights(pretrained)
        self.backbone.init_weights()
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()
        self.rpn_head.init_weights()
        # self.shared_head.init_weights(pretrained=pretrained)
        # self.shared_head_rbbox.init_weights(pretrained=pretrained)
        self.bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()
        self.rbbox_roi_extractor.init_weights()
        self.rbbox_head.init_weights()

    def execute_train(self, images, targets=None):
        x = self.backbone(images)
        x = self.neck(x)
        losses = dict()

        # extract targets
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

        # RPN forward and loss
        rpn_outs = self.rpn_head(x)
        rpn_loss_inputs = rpn_outs + (gt_bboxes, image_meta, self.train_cfg.rpn)
        rpn_losses = self.rpn_head.loss(*rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        proposal_inputs = rpn_outs + (image_meta, proposal_cfg)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        # assign gts and sample proposals (hbb assign)
        bbox_assigner = build_from_cfg(self.train_cfg.rcnn[0].assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn[0].sampler, BOXES)
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # bbox head forward and loss
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        # bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        rbbox_targets = self.bbox_head.get_target(
            sampling_results, gt_obbs, gt_labels, self.train_cfg.rcnn[0])

        loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, *rbbox_targets)
        for name, value in loss_bbox.items():
            losses['s{}.{}'.format(0, name)] = (value)

        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        roi_labels = rbbox_targets[0]
        with jt.no_grad():
            rotated_proposal_list = self.bbox_head.refine_rbboxes(
                roi2droi(rois), roi_labels, bbox_pred, pos_is_gts, image_meta)

        # assign gts and sample proposals (rbb assign)
        bbox_assigner = build_from_cfg(self.train_cfg.rcnn[1].assigner, BOXES)
        bbox_sampler = build_from_cfg(self.train_cfg.rcnn[1].sampler, BOXES)
        num_imgs = images.shape[0]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            gt_obbs_best_roi = jt.array(choose_best_Rroi_batch(gt_obbs[i]))
            assign_result = bbox_assigner.assign(
                rotated_proposal_list[i], gt_obbs_best_roi, gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                rotated_proposal_list[i],
                gt_obbs_best_roi,
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        # (batch_ind, x_ctr, y_ctr, w, h, angle)
        rrois = dbbox2roi([res.bboxes for res in sampling_results])
        # feat enlarge
        rrois[:, 3] = rrois[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois[:, 4] = rrois[:, 4] * self.rbbox_roi_extractor.h_enlarge
        rbbox_feats = self.rbbox_roi_extractor(x[:self.rbbox_roi_extractor.num_inputs], rrois)
        # rbbox_feats = self.shared_head_rbbox(rbbox_feats)
        cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        rbbox_targets = self.rbbox_head.get_target_rbbox(sampling_results, gt_obbs, gt_labels,
                                                            self.train_cfg.rcnn[1])
        loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred, *rbbox_targets)
        for name, value in loss_rbbox.items():
            losses['s{}.{}'.format(1, name)] = (value)
        return losses

    def execute_test(self, images, targets, rescale=False):
        # extract targets
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
        x = self.neck(x)
        rpn_outs = self.rpn_head(x)
        proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
        proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        rois = bbox2roi(proposal_list)
        bbox_feats = self.bbox_roi_extractor(x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        # bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_label = jt.argmax(cls_score, dim=1)[0]
        rrois = self.bbox_head.regress_by_class_rbbox(roi2droi(rois), bbox_label, bbox_pred, img_meta[0])

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rbbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rbbox_roi_extractor.h_enlarge

        rbbox_feats = self.rbbox_roi_extractor(x[:len(self.rbbox_roi_extractor.featmap_strides)], rrois_enlarge)
        # rbbox_feats = self.shared_head_rbbox(rbbox_feats)

        rcls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
        det_rbboxes, det_labels = self.rbbox_head.get_det_rbboxes(
            rrois,
            rcls_score,
            rbbox_pred,
            img_meta[0]['img_shape'],
            img_meta[0]['scale_factor'],
            rescale=rescale,
            cfg=self.test_cfg.rcnn)
        rbbox_results = dbbox2result(det_rbboxes, det_labels,
                                     self.rbbox_head.num_classes)

        return [rbbox_results]

    def execute(self, images, targets=None):
        if self.is_training():
            return self.execute_train(images, targets)
        else:
            return self.execute_test(images, targets)
    
    def train(self):
        super(ReDet, self).train()
        for v in self.__dict__.values():
            if isinstance(v, nn.Module):
                v.train()
