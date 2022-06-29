# model settings
input_size = 300
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1])
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SSD_VGG16',
        input_size=input_size,
        pretrained='weights/vgg16_caffe_numpy.pkl'),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    roi_heads=dict(
        type='SSDHead',
        num_classes=80,
        in_channels=[512, 1024, 512, 256, 256, 256],
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder_cfg=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(0., 0., 0., 0.),
            target_stds=(0.1, 0.1, 0.2, 0.2)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.,
                ignore_iof_thr=-1,
                gt_max_assign_all=False),
            smoothl1_beta=1.,
            allowed_border=-1,
            pos_weight=-1,
            neg_pos_ratio=3,
            debug=False),
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
        anno_file='/mnt/disk/flowey/dataset/coco/annotations/instances_train2017.json',
        root='/mnt/disk/flowey/dataset/coco/images/train2017/',
        transforms=[
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32./255,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18/360),
            dict(
                type='Expand',
                mean=img_norm_cfg['mean'],
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(
                type="Resize_keep_ratio",
                min_size=300,
                max_size=300,
                keep_ratio=False
            ),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type="Normalize",
                mean=img_norm_cfg['mean'],
                std=img_norm_cfg['std'],
                to_bgr=True,)

        ],
        batch_size=32,
        num_workers=4,
        shuffle=True
    ),
    val=dict(
        type="COCODataset",
        anno_file='/mnt/disk/flowey/dataset/coco/annotations/instances_val2017.json',
        root='/mnt/disk/flowey/dataset/coco/images/val2017/',
        transforms=[
            dict(
                type="Resize_keep_ratio",
                min_size=300,
                max_size=300,
                keep_ratio=False
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_bgr=True,),
        ],
        num_workers=2,
        batch_size=2,
    ),
    test=dict(
        type="COCODataset",
        anno_file='/mnt/disk/flowey/dataset/coco/annotations/instances_val2017.json',
        root='/mnt/disk/flowey/dataset/coco/images/val2017/',
        transforms=[
            dict(
                type="Resize_keep_ratio",
                min_size=300,
                max_size=300,
                keep_ratio=False
            ),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[1, 1, 1],
                to_bgr=True,),
        ],
        num_workers=1,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD',
    lr=2e-3,
    momentum=0.9,
    weight_decay=5e-4)

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    milestones=[45,55])

logger = dict(
    type="RunLogger")

max_epoch = 60
eval_interval = 3
checkpoint_interval = 3
log_interval = 50