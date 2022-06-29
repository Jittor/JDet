_base_ = ['yolo_model_base.py', 'yolo_dataset_base.py', 'yolo_optimizer_base.py', 'yolo_scheduler_base.py']
batch_size = 16
max_epoch = 12
log_interval=10
eval_interval=12
checkpoint_interval = 1
stride=32
imgsz=640
imgsz_test=640
dataset_type = 'YoloDataset'

model = dict(
    type='YOLOv5M',
    ema=True,
    imgsz=imgsz,
    is_coco=True
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
    max_steps=max_epoch,
    warmup_iters=22179
)
dataset = dict(
    val=dict(
        type=dataset_type,
        task='val',
        path='/home/wang/workspace/datasets/coco/val2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test,
        is_coco=True
        ),
    train=dict(
        type=dataset_type,
        task='train',
        path='/home/wang/workspace/datasets/coco/train2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True,
        is_coco=True
        ),
    test=dict(
        type=dataset_type,
        task='test',
        path='/home/wang/workspace/datasets/coco/val2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test,
        is_coco=True
        ),
)

logger = dict(
    type="RunLogger")
