_base_ = 'retinanet_r50v1d_fpn_fair.py'

scheduler = dict(
    warmup_iters= 0)
pretrained_weights="work_dirs/retinanet_29/checkpoints/ckpt_30.pkl"