# Gliding
Gliding with same hyperparameters of [maskrcnn_benchmark](https://github.com/MingtaoFu/gliding_vertex)) is coming soon.
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
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 72.14  | [arxiv](https://arxiv.org/abs/1911.09358)| [config](projects/gliding/configs/gliding_r50_fpn_1x_dota_with_flip.py) | [model]() |
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip+ra90+bc|-|  SGD   |   1x    | 74.94   | [arxiv](https://arxiv.org/abs/1911.09358)| [config](projects/gliding/configs/gliding_r50_fpn_1x_dota_with_flip_rotate_balance_cate.py) | [model]() |
