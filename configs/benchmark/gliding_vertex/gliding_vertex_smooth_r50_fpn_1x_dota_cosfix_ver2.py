_base_ = ['./gliding_vertex_r50_fpn_1x_dota.py']

model = dict(
        bbox_head=dict(
            type='SmoothGlidingRoIHead',
            fix_type='cos',
            fix_coder=dict(type='SmoothGVCoder', pow_iou=2.0, ratio_ver=2),
            loss_fix=dict(loss_weight=1.0),
            loss_ratio=dict(loss_weight=1.0)
        )
)

