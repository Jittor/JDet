## Training
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_obb_r50_fpn_1x_dota.py --task=train
```

## Testing
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_obb_r50_fpn_1x_dota.py --task=test
```

## Performance
mAP on OBB task in DOTA1.0: <b>0.69631387</b>

|    Models     | Dataset | mAP    | Config     | Download   |
| :-----------: | :-----: | :----: | :--------: | :--------: |
| FasterRCNN OBB| DOTA1.0 | 69.63  | [config](configs/faster_rcnn_obb_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/29197095057348d0a392/?dl=1) |

## References
https://github.com/dingjiansw101/aerialdetection