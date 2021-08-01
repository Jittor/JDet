_base_ = 'retinanet_r50v1d_fpn_dota.py'

model = dict(
    rpn_net = dict(
        anchor_generator = dict(
          type= "AnchorGeneratorYangXue",
          strides= [8, 16, 32, 64, 128],
          ratios= [1, 0.5, 2.0, 0.3333333333333333, 3.0, 5.0, 0.2],
          scales= [1, 1.2599210498948732, 1.5874010519681994],
          base_sizes= [32, 64, 128, 256, 512],
          angles= [-90, -75, -60, -45, -30, -15],
          mode= "H",
          yx_base_size= 4.)),
)