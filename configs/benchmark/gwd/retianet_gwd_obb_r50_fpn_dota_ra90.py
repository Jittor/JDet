_base_ = ['./retianet_gwd_obb_r50_fpn_dota.py']

dataset=dict(
    train=dict(
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
            ),
            dict(
                type='RotatedRandomFlip',
                direction="horizontal",
                prob=0.5,
            ),
            dict(
                type='RotatedRandomFlip', 
                direction="vertical",
                prob=0.5,
            ),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
            ),
            dict(
                type = "Pad",
                size_divisor=32,
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,
            )
        ],
    ),
)
