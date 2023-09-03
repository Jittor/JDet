_base_ = ['./s2anet_obb_r50_fpn_1x_dota.py']

dataset=dict(
    train=dict(balance_category=True)
)
