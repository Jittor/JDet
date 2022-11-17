# RetinaNet

### Training
```sh
python run_net.py --config-file=configs/rotated_retinanet_obb_kld_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/rotated_retinanet_obb_kld_r50_fpn_1x_dota.py --task=test
```

### Performance
|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| Rotated-RetinaNet-hbb-kld-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 69.10   | [arxiv](https://arxiv.org/abs/1908.05612)| [config](configs/rotated_retinanet_hbb_r50_fpn_1x_dota.py) | [model]() |
