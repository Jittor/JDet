# YOLOV5

## Training
```sh
python run_net.py --config-file=“configs/yolov5s_coco_12epoch_ema.py” --task=train
python run_net.py --config-file=“configs/yolov5m_coco_12epoch_ema.py” --task=train
python run_net.py --config-file=“configs/yolov5l_coco_12epoch_ema.py” --task=train
python run_net.py --config-file=“configs/yolov5x_coco_12epoch_ema.py” --task=train
```

## Testing
### Testing with trained weights
```sh
python run_net.py --config-file=“configs/yolov5s_coco_12epoch_ema.py” --task=test
python run_net.py --config-file=“configs/yolov5m_coco_12epoch_ema.py” --task=test
python run_net.py --config-file=“configs/yolov5l_coco_12epoch_ema.py” --task=test
python run_net.py --config-file=“configs/yolov5x_coco_12epoch_ema.py” --task=test
```


## Performance
|    Models     | Dataset | mAP@.5:.95 | mAP@.5 | mP | mR | Config     | Download   |
| :-----------: | :-----: | :----: | :--------: | :--------: |
| YOLOV5S| COCO | 19.2 | 34.6 | 44.6 | 37.5 | [config](configs/yolov5s_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/5eaab019296a4ac185e5/?dl=1) |
| YOLOV5M| COCO | 26.2 | 43.3 | 54.6 | 41.9 | [config](configs/yolov5m_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/d89c5cf5e5604802951d/?dl=1) |
| YOLOV5L| COCO | 29.9 | 47.5 | 59.4 | 44.6 | [config](configs/yolov5l_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/ee32894861a342fe90d3/?dl=1) |
| YOLOV5X| COCO | 31.1 | 48.7 | 59.3 | 46.4 | [config](configs/yolov5x_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/20405ba3e1984f889c22/?dl=1) |

## References
https://github.com/ultralytics/yolov5/releases/tag/v5.0