# Using JDet with FAIR
## Data Preparing
Downloading FAIR at http://sw.chreos.org/challenge , and save to `$FAIR_PATH$` as:
```
$FAIR_PATH$
├── train
|     ├──images
|     └──labelXmls
├── val
|     ├──images
|     └──labelXmls
└── test
      └──images
```
## Data Preprocessing
Images in FAIR is relatively big, we need to crop each image into several sub-images before training and testing.
```
cd $JDet_PATH$
```
We can set how the FAIR is preprocessed by editing the `configs/preprocess/fair_preprocess_config.py`:
```python
type='FAIR'
source_fair_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/fair'
convert_tasks=['train','val','test']
source_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/fair_DOTA'
target_dataset_path='/home/cxjyxx_me/workspace/JAD/datasets/FAIR/processed'

# available labels: train, val, test, trainval
tasks=[
    dict(
        label='trainval',
        config=dict(
            subimage_size=600,
            overlap_size=150,
            multi_scale=[1.],
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
We need to set `source_dataset_path` to `$FAIR_PATH$`, and set `target_dataset_path` to `$PROCESSED_FAIR_PATH$`.
Then we can set the cropping paramters through `subimage_size` and `overlap_size`, and set `multi_scale` for multi scale training or testing, the tool will first resize the origin image by different scale fators, and cropping each scaled image by `subimage_size` and `overlap_size`.
Finally, run the following script for preprocessing：
```
python tools/preprocess.py --config-file configs/preprocess/fair_preprocess_config.py
```
For the way of configuring the processed FAIR dataset in the model config file, please refer to `$JDet_PATH$/configs/retinanet_r50v1d_fpn_fair.py`:
```
dataset = dict(
    ...
)
```
## Data Postprocessing
The Runner.test() in JDet will automatically merge results of each sub-images in the test set, and generates zip file of the output results in the `submit_zips` directory. 