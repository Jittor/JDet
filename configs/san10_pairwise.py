model = dict(
    type='SAN',
    sa_type=0,
    layers=[2, 1, 2, 4, 1],
    kernels=[3, 7, 7, 7, 7],
    num_classes=1000,
    loss=dict(
        type='SAMSmoothLoss',
        eps=0.1
    ),
    loss_prepare=False
)

# dataset settings
dataset_type = 'ILSVRCDataset'
dataset = dict(
    imgs_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        images_dir='/home/flowey/dataset/ILSVRC2012/train/',
        transforms=[
            dict(type = "Resize",           ## TODO: implement RandomRotatedCrop
                 min_size = 224,
                 max_size = 224,
            ),
            dict(
                type = "RotatedRandomFlip",
                prob = 0.5,
                direction="horizontal",
            ),
            dict(
                type = "Normalize",         ## unknown normalize
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True),
        ],
        batch_size=2,
        ),
    val=dict(
        type=dataset_type,
        batch_size=128,
        images_dir='/home/flowey/dataset/ILSVRC2012/train/',
        transforms=[
            dict(type = "Resize",
                 min_size = 224,
                 max_size = 224,
            ),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True),
        ],
        ),
    test=dict(
        type="ImageDataset",        
        images_dir='/mnt/disk/lxl/dataset/DOTA_1024/test_split/images/',
        transforms=[
            dict(type = "Resize",
                 min_size = 224,
                 max_size = 224,
            ),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True),
        ],
    )
)
# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0001,
    grad_clip=dict(         ##grad_clip: not sure
        max_norm=35, 
        norm_type=2))

# learning policy
scheduler = dict(
    type='CosineAnnealingLR',
    warmup='linear',        ##warmup: not sure
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    max_steps=100)

logger = dict(
    type="RunLogger")
max_epoch = 100
eval_interval = 25
checkpoint_interval = 10
log_interval = 20