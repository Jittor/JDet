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
        start_level=1,
        add_extra_convs="on_input",
        num_outs=5),
    rpn = dict(
        type = "RPNHead",
        in_channels = 256
    ),
    bbox_head=dict(
        type='GlidingHead',
        num_classes=16,
        in_channels=256,
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
            dict(type='RotatedRandomFlip', prob=0.0),
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
        shuffle=False
    ),
    # val=dict(
    #     type="DOTADataset",
    #     anno_file='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/trainval1024.pkl',
    #     image_dir='/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/',
    #     transforms=[
    #         dict(
    #             type = "Pad",
    #             size_divisor=32),
    #         dict(
    #             type = "Normalize",
    #             mean =  [123.675, 116.28, 103.53],
    #             std = [58.395, 57.12, 57.375]),
    #     ],
    #     batch_size=2,
    #     num_workers=4,
    #     shuffle=False
    # ),
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

# when we the trained model from cshuan, image is rgb
max_epoch = 12
eval_interval = 1
checkpoint_interval = 1
log_interval = 50
work_dir = "/mnt/disk/lxl/JDet/work_dirs/gliding_r50_fpn_1x_dota_bs2_tobgr_steplr"
