# model settings
input_size = 300
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SSD_VGG16',
        input_size=input_size,
        pretrained=True),
    roi_heads=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=[512, 1024, 512, 256, 256, 256],
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder_cfg=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0., 0., 0., 0.),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        train_cfg=dict(
            smoothl1_beta=1.,
            neg_pos_ratio=3),
        test_cfg=dict(
            use_sigmoid_cls=False,
            nms_pre=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0,
            score_thr=0.02,
            max_per_img=200),
    )
)
dataset = dict(
    train=dict(
        type="COCODataset",
        anno_file='/home/gmh/dataset/coco128/detections_train2017.json',
        root='/home/gmh/dataset/coco128/images/train2017/',
        transforms=[
            dict(
                type="Resize",
                min_size=300,
                max_size=300
            ),
            dict(
                type="Pad",
                size_divisor=32),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=True,)

        ],
        batch_size=1,
        num_workers=1,
        shuffle=False
    ),

    test=dict(
        type="COCODataset",
        anno_file='/home/gmh/dataset/coco128/detections_train2017_test.json',
        root='/home/gmh/dataset/coco128/images/train2017/',
        transforms=[
            dict(
                type="Resize",
                min_size=300,
                max_size=300
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                # std=[58.395, 57.12, 57.375],
                std=[1, 1, 1],
                to_bgr=False,),
        ],
        num_workers=1,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD',
    lr=0.0,  # 0.01*(1/8.),
    momentum=0.9,
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35,
        norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[8, 11])

logger = dict(
    type="RunLogger")

max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 50
log_interval = 50
resume_path = "/home/gmh/project/yizhang/JDet/ssd300_coco.pth"
