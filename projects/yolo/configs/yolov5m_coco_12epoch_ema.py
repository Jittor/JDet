batch_size = 16
max_epoch = 12
log_interval=1
eval_interval=12
checkpoint_interval = 1
work_dir='/mnt/disk/wang/JDet/projects/yolo/coco-v5m-12epoch-ema'
stride=32
imgsz=640
imgsz_test=640
nc=80
dataset_type = 'YoloDataset'
hyp='/mnt/disk/wang/JDet/projects/yolo1/data/hyp.scratch.yaml'

model = dict(
    type ='YOLOv5S',
    ch = 3, 
    nc = nc,
    pretrained=False,
    imgsz=imgsz,
    ema=True
)
ema = dict(
    type = 'ModelEMA'
)
parameter_groups_generator = dict(
    type='YoloParameterGroupsGenerator',
    weight_decay=0.0005, #hyp[weight_decay]
    batch_size=batch_size
)
optimizer=dict(
    type='SGD',
    lr=0.01, # hyp[lr0]
    momentum=0.937, #hyp[momentum]
    nesterov=True
)
scheduler=dict(
    type='CosineAnnealingLRGroup',
    max_steps=max_epoch,
    min_lr_ratio=0.2, # hyp[lrf]
    warmup_init_lr_pg=[0., 0., 0.1], #[pg0, pg1, pg2]
    warmup_ratio = 0.,
    warmup_initial_momentum = 0.8, #hyp[warmup_momentum]
    warmup = 'linear',
    warmup_iters= max(1000, 7393 * 3) # max(3 epochs, 1000 iters) 
)
dataset = dict(
    val=dict(
        type=dataset_type,
        task='val',
        path='/mnt/disk/wang/coco/val2017.txt',
        batch_size = batch_size,
        num_workers=4,
        stride=stride,
        imgsz=imgsz_test,
        hyp=hyp
        ),
    train=dict(
        type=dataset_type,
        task='train',
        path='/mnt/disk/wang/coco/train2017.txt',
        batch_size = batch_size,
        num_workers=4,
        stride=stride,
        imgsz=imgsz,
        augment=True,
        hyp=hyp
        )
)

logger = dict(
    type="RunLogger")