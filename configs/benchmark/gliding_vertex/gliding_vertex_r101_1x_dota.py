_base_ = ['./gliding_vertex_r50_fpn_1x_dota.py']

model = dict(backbone=dict(type='Resnet101'))

