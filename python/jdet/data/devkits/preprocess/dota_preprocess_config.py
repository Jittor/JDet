type='DOTA'
source_dataset_path='/mnt/disk/lxl/dataset/DOTA/'
target_dataset_path='/mnt/disk/lxl/dataset/DOTA_test/'

# available labels: train, val, test, trainval
tasks=[
    dict(
        label='trainval',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1., 1.5],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]