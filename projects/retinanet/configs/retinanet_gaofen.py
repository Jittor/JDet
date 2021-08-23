_base_ = 'retinanet_r50v1d_fpn_fair.py'

dataset = dict(
    val=None,
    train=dict(
        dataset_dir="{FAIR_PATH}/processed/trainval_600_150_1.0"
    ),
    test = dict(
        images_dir= "{FAIR_PATH}/processed/test_600_150_1.0/images/"
    )
)