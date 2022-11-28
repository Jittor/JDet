import jittor as jt
from jittor import nn
from jdet.utils.registry import MODELS, build_from_cfg, BACKBONES, HEADS, NECKS
import math
from jittor.nn import grid_sample
import cv2
import numpy as np


@MODELS.register_module()
class H2RBox(nn.Module):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 roi_heads=None,
                 crop_size=(768, 768),
                 padding='reflection'):
        super(H2RBox, self).__init__()
        self.backbone = build_from_cfg(backbone, BACKBONES)
        if neck is not None:
            self.neck = build_from_cfg(neck, NECKS)
        else:
            self.neck = None
        self.bbox_head = build_from_cfg(roi_heads, HEADS)

        self.crop_size = crop_size
        self.padding = padding

    def rotate_crop(self, img, theta=0., size=(768, 768), gt_bboxes=None, padding='reflection'):
        # device = img.device
        n, c, h, w = img.shape
        size_h, size_w = size
        crop_h = (h - size_h) // 2
        crop_w = (w - size_w) // 2
        if theta != 0:
            cosa, sina = math.cos(theta), math.sin(theta)
            tf = jt.array([[cosa, -sina], [sina, cosa]], dtype=jt.float)
            # tf = img.new_tensor([[cosa, -sina], [sina, cosa]], dtype=jt.float)
            x_range = jt.linspace(-1, 1, w)
            y_range = jt.linspace(-1, 1, h)
            y, x = jt.meshgrid(y_range, x_range)
            grid = jt.stack([x, y], -1).unsqueeze(0).expand([n, -1, -1, -1])
            grid = grid.reshape(-1, 2).matmul(tf).view(n, h, w, 2)
            # rotate
            img = grid_sample(img, grid, 'bilinear', padding,
                              align_corners=True)
            if gt_bboxes is not None:
                rot_gt_bboxes = []
                for bboxes in gt_bboxes:
                    xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                    ctr = jt.array([[w / 2, h / 2]])
                    # ctr = tf.new_tensor([[w / 2, h / 2]])
                    xy = (xy - ctr).matmul(tf.T) + ctr
                    a = a + theta
                    rot_gt_bboxes.append(jt.concat([xy, wh, a], dim=-1))
                gt_bboxes = rot_gt_bboxes
        img = img[..., crop_h: crop_h + size_h, crop_w:crop_w + size_w]
        if gt_bboxes is None:
            return img
        else:
            crop_gt_bboxes = []
            for bboxes in gt_bboxes:
                xy, wh, a = bboxes[..., :2], bboxes[..., 2:4], bboxes[..., [4]]
                xy = xy - jt.array([[crop_w, crop_h]])
                # xy = xy - xy.new_tensor([[crop_w, crop_h]])
                crop_gt_bboxes.append(jt.concat([xy, wh, a], dim=-1))
            gt_bboxes = crop_gt_bboxes

            return img, gt_bboxes

    def forward_train(self, images, targets):
        rot = (jt.rand(1) * 2 - 1) * math.pi

        gt_bboxes_batch = [target['rboxes'] for target in targets]
        images1, gt_bboxes_batch = self.rotate_crop(images, 0, self.crop_size, gt_bboxes_batch, self.padding)

        for i, gt_bboxes in enumerate(gt_bboxes_batch):
            targets[i]['rboxes'] = gt_bboxes

        feat1 = self.backbone(images1)
        if self.neck:
            feat1 = self.neck(feat1)
        # cv2.imwrite('/home/sjtu/yx/JDet/input.jpg',
        #             images.permute(0, 2, 3, 1).numpy()[0] * np.array([58.395, 57.12, 57.375])
        #             + np.array([123.675, 116.28, 103.53]))
        images2 = self.rotate_crop(images, rot, self.crop_size, padding=self.padding)
        # cv2.imwrite('/home/sjtu/yx/JDet/output.jpg',
        #             images2.permute(0, 2, 3, 1).numpy()[0] * np.array([58.395, 57.12, 57.375])
        #             + np.array([123.675, 116.28, 103.53]))
        feat2 = self.backbone(images2)
        if self.neck:
            feat2 = self.neck(feat2)

        return self.bbox_head.execute_train(feat1, feat2, rot, targets)

    def forward_test(self, images, targets):
        feat = self.backbone(images)
        if self.neck:
            feat = self.neck(feat)
        outs = self.bbox_head.forward(feat)
        return self.bbox_head.get_bboxes(*outs, targets)

    def execute(self, img, targets):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if 'rboxes' in targets[0]:
            return self.forward_train(img, targets)
        else:
            return self.forward_test(img, targets)
