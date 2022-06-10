_base_ = ['yolo_model_base.py', 'yolo_dataset_base.py', 'yolo_optimizer_base.py', 'yolo_scheduler_base.py']
batch_size = 16
max_epoch = 12
log_interval=10
eval_interval=13
checkpoint_interval = 4
stride=32
imgsz=640
imgsz_test=640
dataset_type = 'YoloDataset'


model = dict(
    type='YOLOv5L',
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
        imgsz=imgsz_test
        ),
    train=dict(
        type=dataset_type,
        task='train',
        path='/home/wang/workspace/datasets/coco/train2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True
        ),
    test=dict(
        type=dataset_type,
        task='test',
        path='/home/wang/workspace/datasets/coco/val2017.txt',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz_test,
        ),
)

logger = dict(
    type="RunLogger")

#resume_path='/home/wang/workspace/JDet/work_dirs/yolov5l_coco_12epoch_ema/checkpoints/ckpt_12.pkl'