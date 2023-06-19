_base_ = ['../_base_/dotav1.py', '../_base_/schedule_1x.py', '../retinanet/retinanet_obb_r50_fpn_1x_dota.py']

model = dict(
    bbox_head=dict(
        type = 'NewRotatedRetinaHeadKFIoU',
        loss_bbox=dict(
            type='KFLoss',
            loss_weight=5.0),
    ),
)
