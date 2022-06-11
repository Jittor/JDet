_base_ = ['yolo_model_base.py', 'yolo_dataset_base.py', 'yolo_optimizer_base.py', 'yolo_scheduler_base.py']
batch_size = 16
max_epoch = 300
test_mode = True
stride=32
imgsz=640
imgsz_test=640
nc=80
pretrained_weights='test_datas_yolo/test_yolo.pkl'
dataset_type = 'YoloDataset'

model = dict(
    type ='YOLOv5S',
    imgsz=imgsz,
    ema=False
)
parameter_groups_generator = dict(
    batch_size=batch_size
)
optimizer=dict(
    type='SGD',
    lr=0.01,
    momentum=0.937, 
    nesterov=True
)
scheduler=dict(
    max_steps=max_epoch,
)

dataset = dict(
    train=dict(
        path='/mnt/disk1/wang/coco128/images/train2017',
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=False
        )
)