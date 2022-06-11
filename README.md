# JDet
## Introduction
JDet is an object detection benchmark based on [Jittor](https://github.com/Jittor/jittor), and mainly focus on aerial image object detection (oriented object detection). 

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

Or use ```PYTHONPATH```: 
You can add ```export PYTHONPATH=$PYTHONPATH:{you_own_path}/JDet/python``` into ```.bashrc```, and run
```shell
source .bashrc
```

## Getting Started

### Datasets
The following datasets are supported in JDet, please check the corresponding document before use. 

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

You can also build your own dataset by convert your datas to DOTA format.
### Config
JDet defines the used model, dataset and training/testing method by `config-file`, please check the [config.md](docs/config.md) to learn how it works.
### Train
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=train
```

### Test
If you want to test the downloaded trained models, please set ```resume_path={you_checkpointspath}``` in the last line of the config file.
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=test
```
### Test on images / Visualization
You can test and visualize results on your own image sets by:
```shell
python tools/run_net.py --config-file=configs/s2anet_r50_fpn_1x_dota.py --task=vis_test
```
You can choose the visualization style you prefer, for more details about visualization, please refer to [visualization.md](docs/visualization.md).
<img src="https://github.com/Jittor/JDet/blob/visualization/docs/images/vis2.jpg?raw=true" alt="Visualization" width="800"/>

### Build a New Project
In this section, we will introduce how to build a new project(model) with JDet.
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
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 74.11   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](configs/s2anet/s2anet_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_bs2_steplr_3%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0| 1024/200| flip+ra90+bc|-|  SGD   |   1x    | 76.40   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r50_fpn_1x_dota_rotate_balance.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance%2Fckpt_12.pkl&dl=1) |
| S2ANet-R50-FPN | DOTA1.0|1024/200| flip+ra90+bc+ms |ms|  SGD   |   1x    | 79.72   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r50_fpn_1x_dota_rotate_balance_ms.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r50_fpn_1x_dota_rotate_balance_ms%2Fckpt_12.pkl&dl=1) |
| S2ANet-R101-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 74.28   | [arxiv](https://arxiv.org/abs/2008.09397)| [config](projects/s2anet/configs/s2anet_r101_fpn_1x_dota_bs2.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Fs2anet_r101_fpn_1x_dota_without_torch_pretrained%2Fckpt_12.pkl&dl=1) |
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 72.93  | [arxiv](https://arxiv.org/abs/1911.09358)| [config](projects/gliding/configs/gliding_r50_fpn_1x_dota_with_flip.py) | [model](https://cloud.tsinghua.edu.cn/f/ebeefa1edaf84a4d8a2a/?dl=1) |
| Gliding-R50-FPN |DOTA1.0|1024/200|Flip+ra90+bc|-|  SGD   |   1x    | 74.93   | [arxiv](https://arxiv.org/abs/1911.09358)| [config](projects/gliding/configs/gliding_r50_fpn_1x_dota_with_flip_rotate_balance_cate.py) | [model](https://cloud.tsinghua.edu.cn/f/395ecd3ddaf44bb58ac9/?dl=1) |
| RetinaNet-R50-FPN |DOTA1.0|600/150|-|-|  SGD   |   -    | 62.503   | [arxiv](https://arxiv.org/abs/1708.02002)| [config](configs/retinanet_r50v1d_fpn_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/f12bb566d4be43bfbdc7/) [pretrained](https://cloud.tsinghua.edu.cn/f/6b5db5fdd5304a5abf19/) |
| FasterRCNN-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 69.631   | [arxiv](https://arxiv.org/abs/1506.01497)| [config](configs/faster_rcnn_obb_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/29197095057348d0a392/?dl=1) |
| RoITransformer-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 73.842   | [arxiv](https://arxiv.org/abs/1812.00155)| [config](configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/55fe6380928f4a6582f8/?dl=1) |
| FCOS-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 70.40   | [ICCV19](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.pdf)| [config](configs/fcos_obb_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/d/918bcbf7a10a40fb8dee/files/?p=%2Fmodels%2Ffcos_r50%2Fckpt_12.pkl&dl=1) |
| OrientedRCNN-R50-FPN |DOTA1.0|1024/200|Flip|-|  SGD   |   1x    | 75.62  | [ICCV21](https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf)| [config](configs/oriented_rcnn_r50_fpn_1x_dota_with_flip.py) | [model](https://cloud.tsinghua.edu.cn/f/a50517f7b8e840949d3f/?dl=1) |


**Notice**:

1. ms: multiscale 
2. flip: random flip
3. ra: rotate aug
4. ra90: rotate aug with angle 90,180,270
5. 1x : 12 epochs
6. bc: balance category
7. mAP: mean Average Precision on DOTA1.0 test set

### Plan of Models
<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>

- :heavy_check_mark: S2ANet
- :heavy_check_mark: Gliding
- :heavy_check_mark: RetinaNet
- :heavy_check_mark: Faster R-CNN
- :heavy_check_mark: SSD
- :heavy_check_mark: ROI Transformer
- :heavy_check_mark: fcos
- :heavy_check_mark: Oriented R-CNN
- :heavy_check_mark: YOLOv5
- :clock3: ReDet
- :clock3: R3Det
- :clock3: Cascade R-CNN
- :heavy_plus_sign: CSL
- :heavy_plus_sign: DCL
- :heavy_plus_sign: GWD
- :heavy_plus_sign: KLD
- :heavy_plus_sign: Double Head OBB
- :heavy_plus_sign: Oriented Reppoints
- :heavy_plus_sign: Guided Anchoring
- :heavy_plus_sign: ...

### Plan of Datasets
<b>:heavy_check_mark:Supported  :clock3:Doing :heavy_plus_sign:TODO</b>

- :heavy_check_mark: DOTA1.0
- :heavy_check_mark: DOTA1.5
- :heavy_check_mark: DOTA2.0
- :heavy_check_mark: SSDD
- :heavy_check_mark: SSDD+
- :heavy_check_mark: FAIR
- :heavy_check_mark: COCO
- :heavy_plus_sign: LS-SSDD
- :heavy_plus_sign: DIOR-R
- :heavy_plus_sign: HRSC2016
- :heavy_plus_sign: ICDAR2015
- :heavy_plus_sign: ICDAR2017 MLT
- :heavy_plus_sign: UCAS-AOD
- :heavy_plus_sign: FDDB
- :heavy_plus_sign: OHD-SJTU
- :heavy_plus_sign: MSRA-TD500
- :heavy_plus_sign: Total-Text
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
8. [oriented_rcnn](https://github.com/jbwang1997/OBBDetection/tree/master/configs/obb/oriented_rcnn)
9. [r3det](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection)
10. [AerialDetection](https://github.com/dingjiansw101/AerialDetection)
11. [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
12. [OBBDetection](https://github.com/jbwang1997/OBBDetection)


