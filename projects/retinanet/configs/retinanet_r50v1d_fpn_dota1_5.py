model = dict(
    type = "RetinaNet",
    backbone = dict(
        type = "Resnet50_v1d",
        return_stages =  ["layer1","layer2","layer3","layer4"],
        pretrained = True,
        ),
    neck = dict(
        type= "FPN",
        in_channels= [256,512,1024,2048],
        out_channels= 256,
        start_level= 1,
        add_extra_convs= "on_output",
        num_outs= 5,
        upsample_cfg = dict(
            mode= "bilinear",
            tf_mode= True),
        upsample_div_factor= 2,
        relu_before_extra_convs= True),
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
          ratios= [1, 0.5, 2.0, 0.3333333333333333, 3.0, 5.0, 0.2],
          scales= [1, 1.2599210498948732, 1.5874010519681994],
          base_sizes= [32, 64, 128, 256, 512],
          angles= [-90, -75, -60, -45, -30, -15],
          mode= "H")),
)
dataset = dict(
    val=dict(
        type="DOTADataset",
        version='1_5',
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA1_5/trainval_600_150_1.0',
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
        batch_size=4,
        num_workers=4,
        shuffle=False
    ),
    train=dict(
        type="DOTADataset",
        version='1_5',
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA1_5/trainval_600_150_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=800,
                max_size=800
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
        batch_size=3, 
        num_workers=4,
        shuffle= True
    ),
    test = dict(
      type= "ImageDataset",
      dataset_type="DOTA1_5",
      images_dir= "/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA1_5/test_600_150_1.0/images/",
      transforms= [
        dict(
          type= "RotatedResize",
          min_size= 800,
          max_size= 800),
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
    lr=3 * 5e-4,
    momentum=0.9, 
    weight_decay=1e-4,
    grad_clip=dict(
        max_norm=30.,  # 10*batch_size
        norm_type=2))

scheduler = dict(
    type= "StepLR",
    warmup= "linear",
    warmup_iters= 14000,
    warmup_ratio= 0.1,
    milestones= [27])

logger = dict(
    type= "RunLogger")

max_epoch = 30
eval_interval = 10
log_interval = 50
checkpoint_interval = 1
pretrained_weights="weights/yx_init_pretrained.pk_jt.pk"
merge_nms_threshold_type = 1

parameter_groups_generator = dict(
    type = "YangXuePrameterGroupsGenerator",
    conv_bias_grad_muyilpy = 2.0,
    conv_bias_weight_decay = 0.,
    freeze_prefix=['backbone.C1']
)