model = dict(
    type = "RetinaNet",
    backbone = dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages =  ["layer1","layer2","layer3","layer4"],
        pretrained = True,
        ),
    neck = dict(
        type= "FPN",
        in_channels= [256,512,1024,2048],
        out_channels= 256,
        start_level= 1,
        add_extra_convs= "on_input",
        num_outs= 5),
    rpn_net = dict(
        type= "RetinaHead",
        n_class= 15,
        in_channels= 256,
        stacked_convs= 4,
        mode= "R",
        score_threshold= 0.05,
        nms_iou_threshold= 0.3,
        max_dets= 10000,
        roi_beta= 1 / 9.,
        cls_loss_weight= 1.,
        loc_loss_weight= 0.2,

        anchor_generator = dict(
          type= "AnchorGeneratorRotated",
          strides= [8, 16, 32, 64, 128],
          ratios= [0.5, 1.0, 2.0],
          scales= [4., 5.0396842, 6.34960421],
          mode= "H")),
)
dataset = dict(
    train=dict(
        type="DOTADataset",
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict( #TODO:vertical
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
      images_dir= "/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images/",
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
    type='GradMutilpySGD', 
    lr=0.005,
    momentum=0.9, 
    weight_decay=1e-4,
    grad_clip=dict(
        max_norm=35.,
        norm_type=2))

scheduler = dict(
    type= "StepLR",
    warmup= "linear",
    warmup_iters= 500,
    warmup_ratio= 0.001,
    milestones= [8,11])

logger = dict(
    type= "RunLogger")

max_epoch = 12
eval_interval = 10
log_interval = 50
checkpoint_interval = 1
pretrained_weights="weights/obb_epoch_1.pk"
merge_nms_threshold_type = 1