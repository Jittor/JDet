# Localization Distillation on Rotated RetinaNet

paper: [CVPR 2022](https://arxiv.org/abs/2102.12252), [Journal extension](https://arxiv.org/abs/2204.05957)

### Training a teacher model

```sh
python run_net.py --config-file=configs/ld/rotated_retinanet_obb_distribution_r50_fpn_1x_dota.py --task=train
```
Or download a pretrained teacher model [rotated_retinanet_obb_distribution_r50_fpn_1x_dota](https://cloud.tsinghua.edu.cn/f/b737fe43de8c47a6810e/?dl=1)

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

|    Method     | Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | AP50 | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: | :--------: |
| Original |600/150| flip|-|  SGD   |   1x    | 67.2 | [config](configs/rotated_retinanet_obb_r18_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/5b7825e148024e38b57d/?dl=1) |
| box distribution | 600/150| flip|-|  SGD   |   1x    | 68.5 | [config](configs/rotated_retinanet_obb_distribution_r18_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/43000d3adc1349138632/?dl=1) |
| LD + KD | 600/150| flip|-|  SGD   |   1x    | 69.8 | [config](configs/ld_rotated_retinanet_obb_r18_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/0f3f65c1e7b5401cb5b3/?dl=1) |
