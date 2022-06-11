# YOLOV5

## Dataset
YOLO dataset takes in a path to a txt file that contains all the paths to the training/validating/testing images separated by lines. 
By default, YOLO dataset will expect the labels to be contained in the __labels__ folder next to the __images__ folder containing all the images. Both of these folders will have subdirectories seprating the training, validating, and testing images. The labels are stored in YOLO txt format. 
To use the datasets, simply change the path parameter in the config files to the respective __train/val/test.txt__ files. 
Example:
```sh
dataset---images---train---000000009.jpg
        |         |        |-000000010.jpg
        |         |        .
        |         |        .
        |         |        
        |         |
        |         |-val----000000009.jpg
        |                |-000000010.jpg
        |                .
        |                .
        |                
        |-labels---train---000000009.txt
        |         |      |-000000010.txt
        |         |      .
        |         |      .
        |         |      
        |         |
        |         |-val----000000009.txt
        |                |-000000010.txt
        |                .
        |                .
        |
        |-train.txt
        |-val.txt
```                 
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
All the following models are trained on COCO2017 dataset with the scores evaluated on the COCO2017 validation dataset. 
The models are trained for 12 epochs with default settings. Click the links in the table to go to the cofig files 
or to download pretrained weights, which can be used by setting the __resume_path__ variable in the respective config files. 

|    Models     | Dataset | mAP@.5:.95 | mAP@.5 | mP | mR | Config     | Download   |
| :-----------: | :-----: | :----: | :--------: | :--------: |
| YOLOV5S| COCO | 19.2 | 34.7 | 48.2 | 35.5 | [config](configs/yolov5s_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/5eaab019296a4ac185e5/?dl=1) |
| YOLOV5M| COCO | 26.1 | 42.9 | 52.6 | 43.0 | [config](configs/yolov5m_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/d89c5cf5e5604802951d/?dl=1) |
| YOLOV5L| COCO | 29.9 | 47.2 | 54.0 | 47.9 | [config](configs/yolov5l_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/ee32894861a342fe90d3/?dl=1) |
| YOLOV5X| COCO | 31.1 | 48.6 | 58.4 | 46.5 | [config](configs/yolov5x_coco_12epoch_ema.py) | [model](https://cloud.tsinghua.edu.cn/f/20405ba3e1984f889c22/?dl=1) |

## References
https://github.com/ultralytics/yolov5/releases/tag/v5.0