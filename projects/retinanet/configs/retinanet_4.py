#for grad_check
_base_ = 'retinanet.py'
work_dir = "./exp/retinanet_4"
pretrained_weights="cache_check/init.pk_jt.pk"

dataset = dict(
    train=dict(
        type="DOTADataset",
        annotations_file='/mnt/disk/cxjyxx_me/JAD/datasets/DOTA_mini/splits/trainval_600_150_mini/trainval.pkl',
        images_dir='/mnt/disk/cxjyxx_me/JAD/datasets/DOTA_mini/splits/trainval_600_150_mini/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=800,
                max_size=800
            ),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        batch_size=1, 
        num_workers=4,
        shuffle= False
    ),
    _cover_=True
)

scheduler = dict(
    type= "StepLR",
    warmup= "linear",
    warmup_iters= 0,
    warmup_ratio= 0.1,
    milestones= [10000])

optimizer = dict(
    lr=500.)

log_interval = 1
max_epoch = 2
checkpoint_interval = 1
