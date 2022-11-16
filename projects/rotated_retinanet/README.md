# S2ANet

### Training
```sh
python run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=test
```
### Performance
|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 74.11   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](configs/s2anet_r50_fpn_1x_dota_bs2_steplr.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_bs2_steplr_3%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0| 1024/200| flip+ra90+bc|-|  SGD   |   1x    | 76.40   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](configs/s2anet_r50_fpn_1x_dota_rotate_balance.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip+ra90+bc+ms |ms|  SGD   |   1x    | 79.72   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](configs/s2anet_r50_fpn_1x_dota_rotate_balance_ms.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance_ms%2Fckpt_12.pkl&dl=1) |
| S2ANet-R101-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 74.28   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](configs/s2anet_r101_fpn_1x_dota_bs2.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r101_fpn_1x_dota_without_torch_pretrained%2Fckpt_12.pkl&dl=1) |
