_base_ = ['yolo_model_base.py', 'yolo_dataset_base.py', 'yolo_optimizer_base.py', 'yolo_scheduler_base.py']
batch_size = 16
max_epoch = 12
log_interval=1000
eval_interval=12
checkpoint_interval = 1
work_dir='/mnt/disk/wang/JDet/projects/yolo/coco-v5s-12epoch-ema'
stride=32
imgsz=640
imgsz_test=640
dataset_type = 'YoloDataset'


model = dict(
    type='YOLOv5S',
    pretrained=False,
    ema=True,
    imgsz=imgsz
)

optimizer=dict(
    type='SGD',
    lr=0.01, # hyp[lr0]
    momentum=0.937, #hyp[momentum]
    nesterov=True
)
parameter_groups_generator = dict(
    batch_size=batch_size
)
scheduler=dict(
    max_steps=max_epoch
)
dataset = dict(
    val=dict(
        type=dataset_type,
        task='val',
        path='/mnt/disk/wang/coco/val2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test
        ),
    train=dict(
        type=dataset_type,
        task='train',
        path='/mnt/disk/wang/coco/train2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True
        ),
    test=dict(
        type=dataset_type,
        task='train',
        path='/mnt/disk/wang/coco/test-dev2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test,
        ),
)

logger = dict(
    type="RunLogger")