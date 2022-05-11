batch_size = 16
max_epoch = 300
test_mode = True
stride=32
imgsz=640
imgsz_test=640
nc=80
pretrained_weights='test_datas_yolo/test_yolo.pkl'
dataset_type = 'YoloDataset'
data_path='data/coco128.yaml'
hyp='data/hyp.scratch.yaml'

model = dict(
    type ='YOLOv5S',
    ch = 3, 
    nc = nc,
    pretrained=False,
    imgsz=imgsz,
    hyp=hyp
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
    lr=0.01,
    momentum=0.937, 
    nesterov=True
)
scheduler=dict(
    type='CosineAnnealingLR',
    max_steps=max_epoch,
    min_lr_ratio=0.2, 
)
dataset = dict(
    train=dict(
        type=dataset_type,
        task='train',
        path=data_path,
        batch_size = batch_size,
        num_workers=8,
        stride=stride,
        imgsz=imgsz,
        augment=True,
        hyp=hyp
        )
)