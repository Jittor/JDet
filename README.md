# JDet
## Introduction
JDet is a object detection benchmark based on [Jittor](https://github.com/Jittor/jittor). 

<!-- **Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++.
-  -->

<!-- Framework details are avaliable in the [framework.md](docs/framework.md) -->
## Install
JDet environment requirements:

* System: **Linux**(e.g. Ubuntu/CentOS/Arch), **macOS**, or **Windows Subsystem of Linux (WSL)**
* Python version >= 3.7
* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0)
* GPU compiler (optional)
    * nvcc (>=10.0 for g++ or >=10.2 for clang)
* GPU library: cudnn-dev (recommend tar file installation, [reference link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar))

**Step 1: Install the requirements**
```shell
git clone https://github.com/Jittor/JDet
cd JDet
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JDet**
 
```shell
cd JDet
# suggest this 
python setup.py develop
# or
python setup.py install
```
If you don't have permission for install,please add ```--user```.

Or use ```PYTHONPATH```
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JDet/python``` into ```.bashrc```
```shell
source .bashrc
```

## Getting Started

### Data
DOTA Dataset documents are avaliable in the [dota.md](docs/dota.md)

FAIR Dataset documents are avaliable in the [fair.md](docs/fair.md)
### Config
Config documents are avaliable in the [config.md](docs/config.md)
### Train
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=train
```

### Test
If you want to test the downloaded trained models, please set ```resume_path={you_checkpointspath}``` in the last line of the config file.
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=test
```
## Build a New Project
In this document, we will introduce how to build a new project(model) with JDet.
We need to install JDet first, and build a new project by:
```sh
mkdir $PROJECT_PATH$
cd $PROJECT_PATH$
cp $JDet_PATH$/tools/run_net.py ./
mkdir configs
```
Then we can build and edit `configs/base.py` like `$JDet_PATH$/configs/retinanet.py`.
If we need to use a new layer, we can define this layer at `$PROJECT_PATH$/layers.py` and import `layers.py` in `$PROJECT_PATH$/run_net.py`, then we can use this layer in config files.
Then we can train/test this model by:
```sh
python run_net.py --config-file=configs/base.py --task=train
python run_net.py --config-file=configs/base.py --task=test
```

## Models

|    Models     | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 74.11   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r50_fpn_1x_dota_bs2_steplr_3.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_bs2_steplr_3%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0| 1024/200| flip+ra90+bc|-|  SGD   |   1x    | 76.40   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r50_fpn_1x_dota_rotate_balance.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip+ra90+bc+ms |ms|  SGD   |   1x    | 79.72   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r50_fpn_1x_dota_rotate_balance_ms.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance_ms%2Fckpt_12.pkl&dl=1) |
| S2ANet-R101-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 74.28   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r101_fpn_1x_dota_bs2.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r101_fpn_1x_dota_without_torch_pretrained%2Fckpt_12.pkl&dl=1) |
| Gliding-R50-FPN |DOTA1.0|1024/200|flip+ms|ms|  SGD   |   1x    | 67.42   | [arxiv]()| [config](projects/gliding/configs/gliding_r50_fpn_1x_dota_without_rotate_ms.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fgliding_r50_fpn_1x_dota_bs2_tobgr_steplr_norotate_ms%2Fckpt_12.pkl&dl=1) |
| Gliding-R101-FPN |DOTA1.0|1024/200|flip+ms+ra90+bc|ms|  SGD   |   1x    | 69.53   | [arxiv]()| [config](projects/gliding/configs/gliding_r101_fpn_2x_dota_with_rotate_balance_cate_ms.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fgliding_r101_fpn_1x_dota_bs2_tobgr_steplr_rotate_balance_ms%2Fckpt_12.pkl&dl=1) |
| RetinaNet-R50-FPN |DOTA1.0|600/150|-|-|  SGD   |   -    | 62.503   | [arxiv](https://arxiv.org/abs/1708.02002)| [config](configs/retinanet_r50v1d_fpn_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/f12bb566d4be43bfbdc7/) [pretrained](https://cloud.tsinghua.edu.cn/f/6b5db5fdd5304a5abf19/) |


**Notice**:

1. ms: multiscale 
2. flip: random flip
3. ra: rotate aug
4. ra90: rotate aug with angle 90,180,270
5. 1x : 12 epochs
6. bc: balance category

### Plan
<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>

- :heavy_check_mark: S2ANet
- :heavy_check_mark: Gliding
- :heavy_check_mark: RetinaNet
- :heavy_check_mark: Faster R-CNN
- :heavy_check_mark: SSD
- :heavy_check_mark: ROI Transformer
- :clock3: ReDet
- :clock3: YOLOv5
- :clock3: R3Det
- :clock3: Cascade R-CNN
- :clock3: fcos
- :heavy_plus_sign: CSL
- :heavy_plus_sign: DCL
- :heavy_plus_sign: GWD
- :heavy_plus_sign: KLD
- :heavy_plus_sign: ...


## Contact Us


Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

QQ Group: 761222083


<img src="https://cg.cs.tsinghua.edu.cn/jittor/images/news/2020-12-8-21-19-1_2_2/fig4.png" width="200"/>

## The Team


JDet is currently maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in JDet and want to improve it, Please join us!


## Citation


```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [Detectron2](https://github.com/facebookresearch/detectron2)
3. [mmdetection](https://github.com/open-mmlab/mmdetection)
4. [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
5. [RotationDetection](https://github.com/yangxue0827/RotationDetection)
6. [s2anet](https://github.com/csuhan/s2anet)
7. [gliding_vertex](https://github.com/MingtaoFu/gliding_vertex)
8. [r3det](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection)
9. [AerialDetection](https://github.com/dingjiansw101/AerialDetection)
10. [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)


