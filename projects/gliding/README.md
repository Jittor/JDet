# Gliding
Gliding with same hyperparameters of [maskrcnn_benchmark](https://github.com/MingtaoFu/gliding_vertex)) is coming soon.
### Training
```sh
python run_net.py --config-file=configs/gliding_r101_fpn_1x_dota_with_rotate_balance_cate_ms.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/gliding_r101_fpn_1x_dota_with_rotate_balance_cate_ms.py --task=test
```