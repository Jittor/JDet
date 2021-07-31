from jdet.ops.fr import FeatureRefineModule
from jittor import nn,init 
import jittor as jt 
from jdet.utils.registry import build_from_cfg,BACKBONES,HEADS,NECKS

class R3Det(nn.Module):
    """
    Rotated Refinement RetinaNet
    """

    def __init__(self,
                 num_refine_stages,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 refine_heads=None):
        super(R3Det, self).__init__()
        self.num_refine_stages = num_refine_stages
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.bbox_head = build_from_cfg(bbox_head,HEADS)
        self.feat_refine_module = nn.ModuleList()
        self.refine_head = nn.ModuleList()
        for i, (frm_cfg, refine_head) in enumerate(zip(frm_cfgs, refine_heads)):
            self.feat_refine_module.append(FeatureRefineModule(**frm_cfg))
            self.refine_head.append(build_from_cfg(refine_head,HEADS))

    def execute(self,images,targets):
        pass 
        x = self.backbone(images)
        if self.neck:
            x = self.neck(x)

        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        losses = dict()
        x = self.extract_feat(img)

        outs = self.bbox_head(x)

        train_cfg = self.train_cfg['s0']
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        loss_base = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        for name, value in loss_base.items():
            losses['s0.{}'.format(name)] = value

        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            lw = self.train_cfg.stage_loss_weights[i]
            train_cfg = self.train_cfg['sr'][i]

            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            loss_refine = self.refine_head[i].loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
            for name, value in loss_refine.items():
                losses['sr{}.{}'.format(i, name)] = (
                    [v * lw for v in value] if 'loss' in name else value)

            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        return losses

    def simple_test(self,
                    img,
                    img_meta,
                    rescale=False):
        if 'tile_offset' in img_meta[0]:
            # using tile-cropped TTA. force using aug_test instead of simple_test
            return self.aug_test(imgs=[img], img_metas=[img_meta], rescale=True)

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        rois = self.bbox_head.filter_bboxes(*outs)
        # rois: list(indexed by images) of list(indexed by levels)
        for i in range(self.num_refine_stages):
            x_refine = self.feat_refine_module[i](x, rois)
            outs = self.refine_head[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.refine_head[-1].get_bboxes(*bbox_inputs, rois=rois)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.refine_head[-1].num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results[0]

    def aug_test(self, imgs, img_metas, rescale=True):
        AUG_BS = 8
        assert rescale, '''while r3det uses overlapped cropping augmentation by default,
        the result should be rescaled to input images sizes to simplify the test pipeline'''
        if 'tile_offset' in img_metas[0][0]:
            assert imgs[0].size(0) == 1, '''when using cropped tiles augmentation,
            image batch size must be set to 1'''
            aug_det_bboxes, aug_det_labels = [], []
            num_augs = len(imgs)
            for idx in range(0, num_augs, AUG_BS):
                img = imgs[idx:idx + AUG_BS]
                img_meta = img_metas[idx:idx + AUG_BS]
                act_num_augs = len(img_meta)
                img = torch.cat(img, dim=0)
                img_meta = sum(img_meta, [])
                # for img, img_meta in zip(imgs, img_metas):
                x = self.extract_feat(img)
                outs = self.bbox_head(x)
                rois = self.bbox_head.filter_bboxes(*outs)
                # rois: list(indexed by images) of list(indexed by levels)
                det_bbox_bs = [[] for _ in range(act_num_augs)]
                det_label_bs = [[] for _ in range(act_num_augs)]
                for i in range(self.num_refine_stages):
                    x_refine = self.feat_refine_module[i](x, rois)
                    outs = self.refine_head[i](x_refine)
                    if i + 1 in range(self.num_refine_stages):
                        rois = self.refine_head[i].refine_bboxes(*outs, rois=rois)

                    bbox_inputs = outs + (img_meta, self.test_cfg, False)
                    bbox_bs = self.refine_head[i].get_bboxes(*bbox_inputs, rois=rois)
                    # [(rbbox_aug0, class_aug0), (rbbox_aug1, class_aug1), (rbbox_aug2, class_aug2), ...]
                    for j in range(act_num_augs):
                        det_bbox_bs[j].append(bbox_bs[j][0])
                        det_label_bs[j].append(bbox_bs[j][1])

                for j in range(act_num_augs):
                    det_bbox_bs[j] = torch.cat(det_bbox_bs[j])
                    det_label_bs[j] = torch.cat(det_label_bs[j])

                aug_det_bboxes += det_bbox_bs
                aug_det_labels += det_label_bs

            aug_det_bboxes, aug_det_labels = merge_tiles_aug_rbboxes(
                aug_det_bboxes,
                aug_det_labels,
                img_metas,
                self.test_cfg.merge_cfg,
                self.CLASSES)

            return rbbox2result(aug_det_bboxes, aug_det_labels, self.refine_head[-1].num_classes)

        else:
            raise NotImplementedError