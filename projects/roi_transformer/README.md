## Training
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --task=train
```

## Testing
```sh
python3.7 tools/run_net.py --config-file=configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py --task=test
```

## Performance
configs/faster_rcnn_RoITrans_r50_fpn_1x_dota_old.py
mAP on OBB task in DOTA1.0: <b>0.7356144</b>

configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py
mAP on OBB task in DOTA1.0: <b>0.7469626</b>

## References
https://github.com/dingjiansw101/aerialdetection