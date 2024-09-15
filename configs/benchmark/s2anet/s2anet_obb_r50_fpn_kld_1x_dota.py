_base_ = ['./s2anet_obb_r50_fpn_1x_dota.py']

model = dict(
    bbox_head=dict(
        odm_cfg=dict(
            reg_decoded_bbox=True,
            loss_bbox=dict(
                type='GDLoss_v1',
                loss_type='kld',
                fun='log1p',
                tau=1.0,
                loss_weight=5.5,
            ),
        ),
    ),
)

