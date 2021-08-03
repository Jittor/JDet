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
        type = "RPNHead",
        in_channels = 256,
        num_classes=2,
        min_bbox_size = -1,
        nms_thresh = 0.3,
        nms_pre = 1200,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='CrossEntropyLoss',
            loss_weight=1.0),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False)
    ),
    bbox_head=dict(
        type='GlidingHead',
        num_classes=16,
        in_channels=256,
        representation_dim = 1024,
        pooler_resolution =  7, 
        pooler_scales = [1/4.,1/8., 1/16., 1/32., 1/64.],
        pooler_sampling_ratio = 0,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=2000,
        box_weights = (10., 10., 5., 5.),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        iou_calculator=dict(type='BboxOverlaps2D')),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0)),
        cls_loss=dict(
            type='CrossEntropyLoss',
            ),
        reg_loss=dict(
            type='SmoothL1Loss', 
            beta=1.0 / 9.0, 
            loss_weight=1.0),
        )
    )
dataset = dict(
    train=dict(
        type="DOTADataset",
        annotations_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type='RotatedRandomFlip', 
                prob=0.5),
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
        shuffle=True,
        filter_empty_gt=False
    ),
    val=dict(
        type="DOTADataset",
        annotations_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=2,
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        images_file='/mnt/disk/lxl/dataset/DOTA_1024/test_split/test1024.pkl',
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/test_split/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True,),
        ],
        num_workers=4,
        batch_size=1,
    )
)

optimizer = dict(
    type='SGD', 
    lr=0.01/4.,#0.01*(1/8.), 
    momentum=0.9, 
    weight_decay=0.0001,)
    # grad_clip=dict(
    #     max_norm=35, 
    #     norm_type=2))

scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    milestones=[7, 10])


logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 50
work_dir = "/mnt/disk/lxl/JDet/work_dirs/gliding_r50_fpn_1x_dota_bs2_tobgr_steplr_norotate"
# resume_path = f"{work_dir}/checkpoints/ckpt_12.pkl"
