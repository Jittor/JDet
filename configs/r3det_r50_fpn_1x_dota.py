# model settings
model = dict(
    type='R3Det',
    backbone=dict(
        type='ResNet50',
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        pretrained=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RRetinaHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        use_h_gt=True,
        feat_channels=256,
        anchor_generator=dict(
            type='RAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0, 1.0 / 3.0, 3.0, 0.2, 5.0],
            angles=None,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHABBoxCoder',
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss',
            beta=0.11,
            loss_weight=1.0)),
    frm_cfgs=[
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128]),
        dict(
            in_channels=256,
            featmap_strides=[8, 16, 32, 64, 128])
    ],
    num_refine_stages=2,
    refine_heads=[
        dict(
            type='RRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
        dict(
            type='RRetinaRefineHead',
            num_classes=15,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='PseudoAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHABBoxCoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss',
                beta=0.11,
                loss_weight=1.0)),
    ]
)
# training and testing settings
train_cfg = dict(
    s0=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    sr=[
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.6,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.6,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        )
    ],
    stage_loss_weights=[1.0, 1.0]
)

merge_nms_iou_thr_dict = {
    'roundabout': 0.1, 'tennis-court': 0.3, 'swimming-pool': 0.1, 'storage-tank': 0.1,
    'soccer-ball-field': 0.3, 'small-vehicle': 0.05, 'ship': 0.05, 'plane': 0.3,
    'large-vehicle': 0.05, 'helicopter': 0.2, 'harbor': 0.0001, 'ground-track-field': 0.3,
    'bridge': 0.0001, 'basketball-court': 0.3, 'baseball-diamond': 0.3
}

merge_cfg = dict(
    nms_pre=2000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=merge_nms_iou_thr_dict),
    max_per_img=1000,
)

test_cfg = dict(
    nms_pre=1000,
    score_thr=0.1,
    nms=dict(type='rnms', iou_thr=0.05),
    max_per_img=100,
    merge_cfg=merge_cfg
)