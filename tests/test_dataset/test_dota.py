from jdet.utils.registry import build_from_cfg,DATASETS

dataset = build_from_cfg(dict(
        type="DOTADataset",
        dataset_dir='/home/cxjyxx_me/workspace/JAD/datasets/processed_DOTA/trainval_1024_200_1.0',
        transforms=[
            dict(
                type="RotatedResize",
                min_size=1024,
                max_size=1024
            ),
            dict(
                type='RotatedRandomFlip', 
                prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=True,)
            
        ],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=False,
        balance_category=True
    ),DATASETS)
