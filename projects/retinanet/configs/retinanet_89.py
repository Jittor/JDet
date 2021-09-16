_base_ = 'retinanet_r50v1d_fpn_dota.py'

dataset = dict(
    val=dict(
        annotations_file='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/labels.pkl',
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/images/',
    ),
    train=dict(
        annotations_file='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/labels.pkl',
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_600_150_1.0/images/',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=800,
                max_size=800
            ),
            dict(
                type='RotatedRandomFlip', 
                prob=0.5,
                direction='horizontal'),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        shuffle=False
    ),
    test = dict(
      images_dir= "/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_600_150_1.0/images/",
      ))

flip_test=['H','V']