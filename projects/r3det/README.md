## R3Det
> [R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object](https://arxiv.org/pdf/1908.05612.pdf)

<!-- [ALGORITHM] -->
### Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/r3det.png" width="800"/>
</div>

Rotation detection is a challenging task due to the difficulties of locating the multi-angle objects and separating them effectively from the background. Though considerable progress has been made, for practical settings, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end refined single-stage rotation detector for fast and accurate object detection by using a progressive regression approach from coarse to fine granularity. Considering the shortcoming of feature misalignment in existing refined single stage detector, we design a feature refinement module to improve detection performance by getting more accurate features. The key idea of feature refinement module is to re-encode the position information of the current refined bounding box to the corresponding feature points through pixel-wise feature interpolation to realize feature reconstruction and alignment. For more accurate rotation estimation, an approximate SkewIoU loss is proposed to solve the problem that the calculation of SkewIoU is not derivable. Experiments on three popular remote sensing public datasets DOTA, HRSC2016, UCAS-AOD as well as one scene text dataset ICDAR2015 show the effectiveness of our approach.

### Training
```sh
python run_net.py --config-file=configs/projects/r3det/r3det_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/projects/r3det/r3det_r50_fpn_1x_dota.py --task=test
```

### Performance
|   Models    | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper |                             Config                             | Download   |
|:-----------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------------------------------------------------------------:| :--------: |
| R3Det-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 64.41 | [arxiv](https://arxiv.org/pdf/1908.05612.pdf)| [config](configs/projects/r3det/r3det_r50_fpn_1x_dota.py) | [model]() |

### Citation
```
@inproceedings{yang2021r3det,
    title={R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object},
    author={Yang, Xue and Yan, Junchi and Feng, Ziming and He, Tao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={35},
    number={4},
    pages={3163--3171},
    year={2021}
}
```