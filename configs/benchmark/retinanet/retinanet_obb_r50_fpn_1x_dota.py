_base_ = ['../_base_/dotav1.py', '../_base_/schedule_1x.py']

model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    bbox_head=dict(
        type='NewRotatedRetinaHead',
        num_classes=16,
        in_channels=256,
        feat_channels=256,
        stacked_convs=4,
        bbox_type='obb',
        reg_dim=5,
        background_label=0,
        pos_weight=-1,
        anchor_generator=dict(
            type='AnchorGeneratorXYWHARetinaNet',
            scales=None,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128],
            octave_base_scale=4,
            scales_per_octave=3,
            center_offset=0.5,
        ),
        bbox_coder=dict(
            type='DeltaXYWHABBoxCoder',
            target_means=(0., 0., 0., 0., 0.),
            target_stds=(1., 1., 1., 1., 1.),
            clip_border=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssignerFixMem',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
        cfg=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_thr=0.1),
            max_per_img=2000,
            rescale=True),
    ),
)
