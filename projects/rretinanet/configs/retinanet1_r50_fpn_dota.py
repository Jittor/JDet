model = dict(
    type = "RetinaNet",
    backbone = dict(
        type = "Resnet50",
        return_stages = ["layer1", "layer2", "layer3","layer4"],
        pretrained = True,
        frozen_stages=1
    ),
    neck = dict(
        type = "FPN",
        in_channels= [256,512,1024,2048],
        out_channels= 256,
        start_level= 1,
        add_extra_convs= "on_input",
        num_outs= 5,
    ),
    rpn_net = dict(
        type = "RetinaHead1",
        num_classes = 15,
        in_channels = 256,
        stacked_convs = 4,
        feat_channels= 256,
        # anchor_generator = dict(
        #     type= "Theta0AnchorGenerator",
        #     strides= [8, 16, 32, 64, 128],
        #     ratios= [1, 0.5, 2.0],
        #     octave_base_scale=4,
        #     scales_per_octave=3),
        anchor_generator = dict(
            type= "AnchorGeneratorRotated",
            strides= [8, 16, 32, 64, 128],
            ratios= [1, 0.5, 2.0],
            scales= [4, 4.756828460010884, 5.656854249492381],
            mode="R"
        ),
        bbox_coder=dict(
            type="DeltaXYWHABBoxCoder",
            target_means=[.0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls = dict(
            type= "FocalLoss",
            use_sigmoid=True,
            gamma = 2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D_rotated')
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        test_cfg = dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='obb_nms', iou_thr=0.1),
            max_per_img=2000
        )
    ),

)

dataset = dict(
    val=dict(
        type="DOTADataset",
        annotations_file='/mnt/disk/cxjyxx_me/JAD/datasets/processed_DOTA/trainval_1024_200_1.0/labels.pkl',
        images_dir='/mnt/disk/cxjyxx_me/JAD/datasets/processed_DOTA/trainval_1024_200_1.0/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
        ],
        batch_size=2,
        num_workers=4,
        shuffle=False
    ),
    train=dict(
        type="DOTADataset",
        annotations_file='/mnt/disk/cxjyxx_me/JAD/datasets/processed_DOTA/trainval_1024_200_1.0/labels.pkl',
        images_dir='/mnt/disk/cxjyxx_me/JAD/datasets/processed_DOTA/trainval_1024_200_1.0/images/',
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
    ),
    test = dict(
      type= "ImageDataset",
      images_dir = "/mnt/disk/cxjyxx_me/JAD/datasets/processed_DOTA/trainval_1024_200_1.0/images/",
      transforms= [
        dict(
          type= "RotatedResize",
          min_size= 1024,
          max_size= 1024),
        dict(
          type= "Normalize",
          mean=  [123.675, 116.28, 103.53],
          std= [58.395, 57.12, 57.375],
          to_bgr= False)
      ],
      num_workers= 4,
      batch_size= 32))

optimizer = dict(
    type='SGD', 
    lr=2.5e-3,
    momentum=0.9, 
    weight_decay=1e-4,
    grad_clip=dict(
        max_norm=35,  # 10*batch_size
        norm_type=2))

scheduler = dict(
    type= "StepLR",
    warmup= "linear",
    warmup_iters= 500,
    warmup_ratio= 0.001,
    milestones= [8,11])

logger = dict(
    type= "RunLogger")

work_dir ="./work_dirs/retinanet1"

max_epoch = 12
eval_interval = 1
log_interval = 50
checkpoint_interval = 1
# pretrained_weights="weights/yx_init_pretrained.pk_jt.pk"
# merge_nms_threshold_type = 1

