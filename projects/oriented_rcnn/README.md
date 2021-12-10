# OrientedRCNN
### Training
```sh
python run_net.py --config-file=configs/oriented_rcnn_r50_fpn_1x_dota_with_flip.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/oriented_rcnn_r50_fpn_1x_dota_with_flip.py --task=test
```

## Models

|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| OrientedRCNN-R50-FPN |DOTA1.0|1024/200|flip|-|  SGD   |   1x    | 75.62  | [ICCV21](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)| [config](configs/oriented_rcnn_r50_fpn_1x_dota_with_flip.py) | [model](https://cloud.tsinghua.edu.cn/f/a50517f7b8e840949d3f/?dl=1) |
| OrientedRCNN-R50-FPN |DOTA1.0|1024/200|flip+rr+bc|-|  SGD   |   1x    | 77.76  | [ICCV21](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)| [config](configs/oriented_rcnn_r50_fpn_1x_dota_with_flip_rotate_balance_cate.py) | [model](https://cloud.tsinghua.edu.cn/f/df64dd4980f84433a592/?dl=1) |