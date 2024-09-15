_base_ = ['./retianet_kfiou_obb_r50_fpn_dota.py']

model = dict(backbone=dict(type='Resnet101'))

