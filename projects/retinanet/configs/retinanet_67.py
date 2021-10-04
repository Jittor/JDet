_base_ = 'retinanet_r50v1d_fpn_dota.py'

dataset = dict(
    val=dict(
        annotations_file='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/labels.pkl',
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/images/',
        shuffle=False
    ),
    train=dict(
        annotations_file='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/labels.pkl',
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/images/',
        shuffle=False
    ),
    test = dict(
      images_dir= "/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_600_150_1.0/images/",
      shuffle=False
    ))