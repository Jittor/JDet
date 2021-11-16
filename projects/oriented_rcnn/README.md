# Gliding
### Training
```sh
python run_net.py --config-file=configs/gliding_r50_fpn_1x_dota_with_rotate_balance_cate.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/gliding_r50_fpn_1x_dota_with_rotate_balance_cate.py --task=test
```

## Models

|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 72.93  | [arxiv](https://arxiv.org/abs/1911.09358)| [config](configs/gliding_r50_fpn_1x_dota_with_flip.py) | [model](https://cloud.tsinghua.edu.cn/f/ebeefa1edaf84a4d8a2a/?dl=1) |
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip+ra90+bc|-|  SGD   |   1x    | 74.93   | [arxiv](https://arxiv.org/abs/1911.09358)| [config](configs/gliding_r50_fpn_1x_dota_with_flip_rotate_balance_cate.py) | [model](https://cloud.tsinghua.edu.cn/f/395ecd3ddaf44bb58ac9/?dl=1) |
