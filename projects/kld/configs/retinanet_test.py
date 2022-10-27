# test
_base_ = 'retinanet.py'
pretrained_weights="test_datas_kld/yx_init_pretrained.pk_jt.pk"
max_epoch = 30
test_mode = True

scheduler = dict(
    milestones= [27])

optimizer = dict(
    lr=3 * 5e-2
)

dataset = dict(
    train=dict(
        type="DOTADataset",
        dataset_dir="/home/songxiufeng/workspace/code/JDetPrj/data/preprocessed_DOTA/DOTA1.0/1024_200/trainval_1024_200_1.0",
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type='RotatedRandomFlip', 
                prob=0.5,
                direction='horizontal'),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        batch_size=2, 
        num_workers=4,
        shuffle= True
    ),)

model = dict(
    rpn_net = dict(
        type= "RetinaHead",
        n_class= 15,
        in_channels= 256,
        stacked_convs= 4,
        mode= "R",
        score_threshold= 0.3,
        nms_iou_threshold= 0.3,
        max_dets= 10000,
        cls_loss_weight= 1.,
        loc_loss_weight= 1.,
        loc_loss=dict(
            type='GDLoss_v1',
            loss_type='kld',
            fun='log1p',
            tau=1.0,
            reduction='sum',
            loss_weight=5.5),
        cls_loss=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            loss_weight=1.0),
        anchor_generator = dict(
            type= "AnchorGeneratorRotated",
            strides= [8, 16, 32, 64, 128],
            ratios= [1, 0.5, 2.0, 0.3333333333333333, 3.0, 5.0, 0.2],
            scales= [1, 1.2599210498948732, 1.5874010519681994],
            base_sizes= [32, 64, 128, 256, 512],
            angles= [-90, -75, -60, -45, -30, -15],
            mode= "H"),
        reg_decoded_bbox=True,),
)