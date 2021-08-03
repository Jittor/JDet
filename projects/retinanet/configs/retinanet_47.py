_base_ = 'retinanet_r50v1d_fpn_fair.py'

scheduler = dict(
    warmup_iters= 0)