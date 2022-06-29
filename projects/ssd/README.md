# SSD

## Preparing
<a href="https://cloud.tsinghua.edu.cn/f/d8077ff8c4654bbd8902/?dl=1">Download</a> pretrained weight to weights/vgg16_caffe_numpy.pkl

## Training
```sh
python run_net.py --config-file=configs/ssd300_coco.py --task=train
```

## Testing
### Testing with trained weights
```sh
python run_net.py --config-file=configs/ssd300_coco.py --task=test
```
### Testing with proposed weights
<a href="https://drive.google.com/file/d/1kEF72Ufc2hM_u1ex06XZt2U901J49QZf/view?usp=sharing">Download</a> the trained weights to `$CKPT_PATH$/ssd_ckpt_57.pkl`.
Add following code to the last line of `configs/ssd300_coco.py`
```python
resume_path=$CKPT_PATH$/ssd_ckpt_57.pkl
```
And run:
```sh
python run_net.py --config-file=configs/ssd300_coco.py --task=test
```

## Performance
mAP on Detection task in COCO: <b>0.251</b>

## References
https://github.com/open-mmlab/mmdetection