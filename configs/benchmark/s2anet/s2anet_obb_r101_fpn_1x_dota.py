_base_ = ['./s2anet_obb_r50_fpn_1x_dota.py']

model = dict(backbone=dict(type='Resnet101'))
