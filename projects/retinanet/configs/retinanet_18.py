_base_ = 'retinanet.py'
work_dir = "./exp/retinanet_18"
pretrained_weights="weights/yx_init_pretrained.pk_jt.pk"
optimizer = dict(
    lr=3 * 5e-4
)
