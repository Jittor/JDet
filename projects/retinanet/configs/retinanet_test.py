# test
_base_ = 'retinanet.py'
pretrained_weights="test_datas_retinanet/yx_init_pretrained.pk_jt.pk"
max_epoch = 30
test_mode = True

scheduler = dict(
    milestones= [27])

optimizer = dict(
    lr=3 * 5e-2
)


dataset = dict(
    train=dict(
        transforms=[
            dict(
                type="RotatedResize",
                min_size=800,
                max_size=800
            ),
            dict(
                type='RotatedRandomFlip', 
                prob=0.0,
                direction='horizontal'),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
    ))

model = dict(
    rpn_net = dict(
        anchor_generator = dict(
          _cover_=True,
          type= "AnchorGeneratorRotated",
          strides= [8, 16, 32, 64, 128],
          ratios= [1, 0.5, 2.0, 0.3333333333333333, 3.0, 5.0, 0.2],
          scales= [1, 1.2599210498948732, 1.5874010519681994],
          base_sizes= [32, 64, 128, 256, 512],
          angles= [-90, -75, -60, -45, -30, -15],
          mode= "H")),
)