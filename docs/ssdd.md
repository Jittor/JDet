# Using JDet with SSDD/SSDD+
Using JDet with SAR Ship Detection Dataset (SSDD/SSDD+).
## Data Preparing
Downloading SSDD at https://github.com/TianwenZhang0825/Official-SSDD, and save to `$SSDD_PATH$` as:
```
$SSDD_PATH$
└── Official-SSDD-OPEN
      ├──...
      ├──BBox_SSDD/voc_style
      |     ├──...
      |     ├──JPEGImages_train
      |     ├──JPEGImages_test
      |     ├──Annotations_train
      |     └──Annotations_test
      └──RBox_SSDD/voc_style
            ├──...
            ├──JPEGImages_train
            ├──JPEGImages_test
            ├──Annotations_train
            └──Annotations_test
```
Only the 8 folders above(JPEGImages_train, JPEGImages_test, Annotations_train, Annotations_test) are used in JDet.
## Data Preprocessing
We need to rescale each image into a consistent size before training and testing.
```
cd $JDet_PATH$
```
We can set how the SSDD/SSDD+ is preprocessed by editing the `configs/preprocess/ssdd_preprocess_config.py` or `configs/preprocess/ssdd_plus_preprocess_config.py`, we use `ssdd_plus_preprocess_config.py` as an example:
```python
type='SSDD+'
resize = 800
source_dataset_path='/home/cxjyxx_me/workspace/JAD/SAR/datasets/Official-SSDD-OPEN/RBox_SSDD/voc_style'
target_dataset_path=f'/home/cxjyxx_me/workspace/JAD/SAR/datasets/processed_SSDD_plus/'
convert_tasks=['test', 'train']
```
We need to set `source_dataset_path` to `$SSDD_PATH$/Official-SSDD-OPEN/RBox_SSDD/voc_style`, and set `target_dataset_path` to `$PROCESSED_SSDD_PLUS_PATH$`.
Then we can set the resize paramters through `resize`.
Finally, run the following script for preprocessing：
```
python tools/preprocess.py --config-file configs/preprocess/ssdd_plus_preprocess_config.py
```
For the way of configuring the processed SSDD/SSDD+ dataset in the model config file, please refer to `$JDet_PATH$/projects/s2anet/configs/s2anet_r50_fpn_1x_ssdd.py`/`$JDet_PATH$/projects/s2anet/configs/s2anet_r50_fpn_1x_ssdd_plus.py`:
```
dataset = dict(
    ...
)
```
Note: we rename the 'test' of SSDD/SSDD+ to 'val'.
## Data Postprocessing
The Runner.val() in JDet will automatically calculate the AP50 of SSDD/SSDD+ test set. 