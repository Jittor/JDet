dataset_type = 'DOTADataset'
dataset = dict(
    train=dict(
        type=dataset_type,
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        version='1',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
            ),
            dict(
                type='RotatedRandomFlip',
                direction="horizontal",
                prob=0.5,
            ),
            dict(
                type='RotatedRandomFlip', 
                direction="vertical",
                prob=0.5,
            ),
            dict(
                type = "Pad",
                size_divisor=32,
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,
            )
        ],
        batch_size=2,
        filter_empty_gt=False,
        shuffle=True,
    ),
    val=dict(
        type=dataset_type,
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        version='1',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
            ),
            dict(
                type = "Pad",
                size_divisor=32,
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,
            ),
        ],
    ),
    test=dict(
        type="ImageDataset",        
        images_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/test_1024_200_1.0/images',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024,
            ),
            dict(
                type = "Pad",
                size_divisor=32,
            ),
            dict(
                type = "Normalize",
                mean = [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,
            ),
        ],
    )
)
