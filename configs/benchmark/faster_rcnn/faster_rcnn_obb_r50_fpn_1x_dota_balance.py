_base_ = ['./faster_rcnn_obb_r50_fpn_1x_dota.py']

dataset=dict(
    train=dict(balance_category=True)
)
