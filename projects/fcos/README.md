# FCOS

### Training
```sh
python run_net.py --config-file=configs/fcos_obb_r50_fpn_1x_dota.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/fcos_obb_r50_fpn_1x_dota.py --task=test
```
### Performance
|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| FCOS-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 70.40   | [iccv19](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf)| [config](configs/fcos_obb_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Ffcos_r50%2Fckpt_12.pkl&dl=1) |

