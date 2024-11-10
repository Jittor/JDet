_base_ = ['./retianet_gwd_obb_r50_fpn_dota.py']

dataset=dict(
    train=dict(balance_category=True)
)
