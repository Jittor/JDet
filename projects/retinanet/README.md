# RetinaNet

## Preparing
<a href="https://cloud.tsinghua.edu.cn/f/6b5db5fdd5304a5abf19/">Download</a> pretrained weight to weights/yx_init_pretrained.pk_jt.pk 

## Training
```sh
python run_net.py --config-file=configs/retinanet_r50v1d_fpn_dota.py --task=train
```

## Testing
### Testing with trained weights
```sh
python run_net.py --config-file=configs/retinanet_r50v1d_fpn_dota.py --task=test
```
### Testing with proposed weights
<a href="https://cloud.tsinghua.edu.cn/f/f12bb566d4be43bfbdc7/">Download</a> the trained weights to `$CKPT_PATH$/ckpt_30.pkl`.
Add following code to the last line of `configs/retinanet_r50v1d_fpn_dota.py`
```python
resume_path=$CKPT_PATH$/ckpt_30.pkl
```
And run:
```sh
python run_net.py --config-file=configs/retinanet_r50v1d_fpn_dota.py --task=test
```
The results will be saved in `submit_zips/retinanet_r50v1d_fpn_dota.zip`.

## Performance
mAP on OBB task in DOTA1.0: <b>0.62503</b>

## References
https://github.com/yangxue0827/RotationDetection