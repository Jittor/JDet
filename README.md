# JDet
## Introduction
JDet is Object Detection System  based on [Jittor](https://github.com/Jittor/jittor). 

**Features**
- Automatic compilation. Our framwork is based on Jittor, which means we don't need to Manual compilation for these code with CUDA and C++.
- 

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
git clone https://github.com/li-xl/JDet
cd JDet
python -m pip install -r requirements.txt
```
If you have any installation problems for Jittor, please refer to [Jittor](https://github.com/Jittor/jittor)

**Step 2: Install JDet**
 
```shell
cd JDet
python setup.py install # for user
# or
python setup.py develop # for developer
```
If you don't have permission for install,please add ```--user```.

Or use ```PYTHONPATH```
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JDet/python``` into ```.bashrc```
```shell
source .bashrc
```

## Getting Started

### Train
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=train
```

### Test
if you want to test the pretrained models,please set ```resume_path={you_checkpointspath}``` in config files.
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

|    Models     | Dataset |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
| :-----------: | :-----: |:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
| S2ANet-R50-FPN | DOTA1.0| Flip|-|  SGD   |   1x    | 74.33   | [arxiv](https://arxiv.org/abs/2008.09397)| [config]() | [model]() [log]() |
| S2ANet-R101-FPN |DOTA1.0|Flip|-|  SGD   |   1x    | 74.28   | [arxiv](https://arxiv.org/abs/2008.09397)| [config]() | [model]() [log]() |
| Gliding-R50-FPN |DOTA1.0|-|-|  SGD   |   1x    | 74.0   | [arxiv]()| [config]() | [model]() [log]() |
| RetinaNet-R50-FPN |DOTA1.0|-|-|  SGD   |   1x    | 74.0   | [arxiv]()| [config]() | [model]() [log]() |
| RetinaNet-R50-FPN |DOTA1.0|-|-|  SGD   |   1x    | 74.0   | [arxiv]()| [config]() | [model]() [log]() |
| RetinaNet-R50-FPN |DOTA1.0|-|-|  SGD   |   1x    | 74.0   | [arxiv]()| [config]() | [model]() [log]() |
| SSD |COCO |-|-|  SGD   |   1x    | 74.0   | [arxiv]()| [config]() | [model]() [log]() |


**Notice**:

1. ms: multiscale 
2. flip: random flip
3. ra: rotate aug
4. ra90: rotate aug with angle 90,180,270
5. 1x : 12 epochs

### Incoming Models
YOLO
R3Det



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
