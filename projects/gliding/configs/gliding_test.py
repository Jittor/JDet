# model settings
model = dict(
    type='GlidingVertex',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained= True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn = dict(
        type = "GlidingRPNHead",
        in_channels = 256,
        num_classes=2,
        min_bbox_size = 0,
        nms_thresh = 0.7,
        nms_pre = 2000,
        nms_post = 2000,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='GVDeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            match_low_quality=True,
            assigned_labels_filled=-1,
            ),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False)
    ),
    bbox_head=dict(
        type='GlidingHead',
        num_classes=15,
        in_channels=256,
        representation_dim = 1024,
        pooler_resolution =  7, 
        pooler_scales = [1/4.,1/8., 1/16., 1/32., 1/64.],
        pooler_sampling_ratio = 0,
        score_thresh=0.05,
        nms_thresh=0.3,
        detections_per_img=2000,
        box_weights = (10., 10., 5., 5.),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1,
            match_low_quality=False,
            assigned_labels_filled=-1,
            iou_calculator=dict(type='BboxOverlaps2D')),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        bbox_coder=dict(
            type='GVDeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        fix_coder=dict(type='GVFixCoder'),
        ratio_coder=dict(type='GVRatioCoder'),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2, version=1),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        cls_loss=dict(
            type='CrossEntropyLoss',
            ),
        bbox_loss=dict(
            type='SmoothL1Loss', 
            beta=1.0, 
            loss_weight=1.0
            ),
        fix_loss=dict(
            type='SmoothL1Loss', 
            beta=1.0 / 3.0, 
            loss_weight=1.0,
            ),
        ratio_loss=dict(
            type='SmoothL1Loss', 
            beta=1.0 / 3.0, 
            loss_weight=16.0
            ),
        with_bbox=True,
        with_shared_head=False,
        start_bbox_type='hbb',
        end_bbox_type='poly',
        with_avg_pool=False,
        pos_weight=-1,
        reg_class_agnostic=False,
        ratio_thr=0.8,
        max_per_img=2000,
        )
    )
dataset = dict(
    train=dict(
        type="DOTADataset",
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            # dict(
            #     type='RotatedRandomFlip', 
            #     prob=0.5),
            # dict(
            #     type="RandomRotateAug",
            #     random_rotate_on=True,
            # ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True,)
            
        ],
        batch_size=2,
        num_workers=4,
        shuffle=False,
        filter_empty_gt=False,
        balance_category=False
    ),
)

optimizer = dict(type='SGD',  lr=0.005, momentum=0.9, weight_decay=0.0001, grad_clip=dict(max_norm=35, norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    milestones=[7, 10])

logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 50