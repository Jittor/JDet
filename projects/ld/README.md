# Localization Distillation on Rotated RetinaNet

paper: [CVPR 2022](https://arxiv.org/abs/2102.12252), [Journal extension](https://arxiv.org/abs/2204.05957)

### Training a teacher model

```sh
python run_net.py --config-file=configs/ld/rotated_retinanet_obb_distribution_r50_fpn_1x_dota.py --task=train
```

### Distilling from a pretrained teacher to student
```sh
python run_net.py --config-file=configs/ld/ld_rotated_retinanet_obb_r18_r50_fpn_1x_dota.py --task=train
```
### Val
```sh
python run_net.py --config-file=configs/ld_rotated_retinanet_obb_r18_r50_fpn_1x_dota.py --task=val
```

### Testing
```sh
python run_net.py --config-file=configs/ld_rotated_retinanet_obb_r18_r50_fpn_1x_dota.py --task=test
```
### Performance

#### Rotated-RetinaNet-obb-R18-FPN-1x, train set: DOTA-1.0 train, test set: DOTA-1.0 val.

|    Method     | Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | AP | AP50 | AP75 | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| Original |600/150| flip|-|  SGD   |   1x    | 37.6 | 67.2 | 33.8 | [config](configs/ld/rotated_retinanet_obb_r18_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/6018635728b942c5beb8/?dl=1) |
| box distribution | 600/150| flip|-|  SGD   |   1x    | 38.1 | 68.5 | 34.0 | [config](configs/ld/rotated_retinanet_obb_distribution_r18_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/24bbb1448d4d4a3ba436/?dl=1) |
| LD + KD | 600/150| flip|-|  SGD   |   1x    | 39.6 | 69.8 | 36.2 | [config](configs/ld/ld_rotated_retinanet_obb_r18_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/24bbb1448d4d4a3ba436/?dl=1) |
