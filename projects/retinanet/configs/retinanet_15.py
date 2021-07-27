_base_ = 'retinanet.py'
work_dir = "./exp/retinanet_15"
pretrained_weights="weights/yx_init_pretrained.pk_jt.pk"
max_epoch = 30

scheduler = dict(
    milestones= [27])
