_base_ = ['./retinanet_obb_r50_fpn_1x_dota.py']

dataset=dict(
    train=dict(dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_500_0.5-1.0-1.5'),
    val=dict(dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_500_0.5-1.0-1.5'),
    test=dict(images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_500_0.5-1.0-1.5/images'),
)
