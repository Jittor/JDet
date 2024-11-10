_base_ = ['../_base_/dotav1.py', '../_base_/schedule_1x.py']

model = dict(
    type='FasterRCNNOBBNew',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=5),
    rpn_head=dict(
        type='NewFasterRCNNHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        bbox_type='hbb',
        reg_dim=4,
        allowed_border=0,
        background_label=0,
        pos_weight=-1,
        detach_proposals=True,
        reg_decoded_bbox=False,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            center_offset=0.5,
        ),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder_v0',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type='WeightCrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssignerFixMem',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            match_low_quality=True,
            iou_calculator=dict(type='BboxOverlaps2D'),
        ),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        cfg = dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=1),
        ),
    bbox_head=dict(
        type='SharedConvFCRoIHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        proposal_type='hbb',
        target_type='obb',
        reg_dim=5,
        pos_weight=-1,
        reg_class_agnostic=True,
        with_avg_pool=False,
        with_cls=True,
        with_reg=True,
        assigner=dict(
            type='MaxIoUAssignerFixMem',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=False,
            iou_calculator=dict(type='BboxOverlaps2D'),
        ),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True,
        ),
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2, version=1),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_coder=dict(
            type='HProposalDeltaXYWHTCoder',
            angle_norm_factor=2,
            target_means=[0., 0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        ),
        loss_cls=dict(
            type='WeightCrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        cfg=dict(
            score_thr = 0.05,
            nms = dict(type='ml_nms_rotated', iou_thr=0.1),
            max_per_img = 2000,
        ),
    ),
)
optimizer = dict(lr=0.005)
