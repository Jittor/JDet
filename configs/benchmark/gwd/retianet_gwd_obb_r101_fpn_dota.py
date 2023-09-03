_base_ = ['./retianet_gwd_obb_r50_fpn_dota.py']

model = dict(backbone=dict(type='Resnet101'))

