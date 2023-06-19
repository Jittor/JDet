_base_ = ['./retianet_kfiou_obb_r50_fpn_dota.py']

dataset=dict(
    train=dict(balance_category=True)
)
