## Training
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --task=train
```

## Testing
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --task=test
```

## Performance
configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py
mAP on OBB task in DOTA1.0: <b>0.7384232</b>

|    Models     | Dataset | mAP    | Config     | Download   |
| :-----------: | :-----: | :----: | :--------: | :--------: |
| FasterRCNN OBB + RoITransformer| DOTA1.0 | 73.84  | [config](configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/55fe6380928f4a6582f8/?dl=1) |

## References
https://github.com/dingjiansw101/aerialdetection