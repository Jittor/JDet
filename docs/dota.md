# Using JDet with DOTA
## Data Preparing
Downloading DOTA at https://captain-whu.github.io/DOTA/ , and save to `$DOTA_PATH$` as:
```
$DOTA_PATH$
├── train
|     ├──images
|     └──labelTxt
├── val
|     ├──images
|     └──labelTxt
└── test
      └──images
```
## Data Preprocessing
Images in DOTA is relatively big, we need to crop each image into several sub-images before training and testing.
```
cd $JDet_PATH$/python/jdet/data/devkits/dota_preprocess
```
We can set how the DOTA is preprocessed by editing the `dota_preprocess_config.py`:
```python
source_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/test/DOTA/'
target_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/test/processed_DOTA/'

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
```
We need to set `source_dataset_path` to `$DOTA_PATH$`, and set `target_dataset_path` to `$PROCESSED_DOTA_PATH$`.
Then we can set the cropping paramters through `subimage_size` and `overlap_size`, and set `multi_scale` for multi scale training or testing, the tool will first resize the origin image by different scale fators, and cropping each scaled image by `subimage_size` and `overlap_size`.
Finally, run the following script for preprocessing：
```
python dota_preprocess.py --config-file dota_preprocess_config.py
```
For the way of configuring the processed DOTA dataset in the model config file, please refer to `$JDet_PATH$/configs/retinanet.py`:
```
dataset = dict(
    ...
)
```
## Data Postprocessing
The Runner.test() in JDet will automatically merge results of each sub-images in the test set, and generates submitable zip files in the `submit_zips` directory. 
We can directly submit this file to https://captain-whu.github.io/DOTA/ .