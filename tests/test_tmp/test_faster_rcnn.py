import torch

from jdet.models.networks import FasterRCNN
from jdet.models.losses import CrossEntropyLoss
from jdet.utils.registry import build_from_cfg, HEADS, LOSSES

backbone=dict(
    type='Resnet50',
    frozen_stages=1,
    return_stages=["layer1","layer2","layer3","layer4"],
    pretrained=True)
neck=dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    #Not sure
    start_level=0,
    add_extra_convs="on_input",
    num_outs=5)
rpn_head=dict(
    type='RPNHead',
    in_channels=256,
    feat_channels=256,
    anchor_scales=[8],
    anchor_ratios=[0.5, 1.0, 2.0],
    anchor_strides=[4, 8, 16, 32, 64],
    target_means=[.0, .0, .0, .0],
    target_stds=[1.0, 1.0, 1.0, 1.0],
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))
bbox_roi_extractor=dict(
    type='SingleRoIExtractor',
    roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2),
    out_channels=256,
    featmap_strides=[4, 8, 16, 32])
bbox_head=dict(
    type='SharedFCBBoxHead',
    num_fcs=2,
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=16,
    target_means=[0., 0., 0., 0.],
    target_stds=[0.1, 0.1, 0.2, 0.2],
    reg_class_agnostic=False,
    loss_cls=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        # score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=1000)
    score_thr = 0.05, nms = dict(type='nms', iou_thr=0.5), max_per_img = 2000)
# soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
model = FasterRCNN(backbone=backbone,
                   rpn_head=rpn_head,
                   bbox_roi_extractor=bbox_roi_extractor,
                   bbox_head=bbox_head,
                   train_cfg=train_cfg,
                   test_cfg=test_cfg,
                   neck=neck)
#print(model)
model.load('fake_rcnn.pth')

