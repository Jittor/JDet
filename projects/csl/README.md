## CSL
> [Arbitrary-Oriented Object Detection with Circular Smooth Label](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40)

<!-- [ALGORITHM] -->
### Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/csl.jpg" width="800"/>
</div>

Arbitrary-oriented object detection has recently attracted increasing attention in vision for their importance
in aerial imagery, scene text, and face etc. In this paper, we show that existing regression-based rotation detectors
suffer the problem of discontinuous boundaries, which is directly caused by angular periodicity or corner ordering.
By a careful study, we find the root cause is that the ideal predictions are beyond the defined range. We design a
new rotation detection baseline, to address the boundary problem by transforming angular prediction from a regression
problem to a classification task with little accuracy loss, whereby high-precision angle classification is devised in
contrast to previous works using coarse-granularity in rotation detection. We also propose a circular smooth label (CSL)
technique to handle the periodicity of the angle and increase the error tolerance to adjacent angles. We further
introduce four window functions in CSL and explore the effect of different window radius sizes on detection performance.
Extensive experiments and visual analysis on two large-scale public datasets for aerial images i.e. DOTA, HRSC2016,
as well as scene text dataset ICDAR2015 and MLT, show the effectiveness of our approach.

### Training
```sh
python run_net.py --config-file=configs/rotated_retinanet_obb_csl_gaussian_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/rotated_retinanet_obb_csl_gaussian_r50_fpn_1x_dota.py --task=test
```

### Performance
|   Models    | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper |                             Config                             | Download   |
|:-----------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------------------------------------------------------------:| :--------: |
| CSL-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 66.99   | [arxiv](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40)| [config](configs/rotated_retinanet_obb_csl_gaussian_r50_fpn_1x_dota.py) | [model]() |

### Citation
```
@inproceedings{yang2020arbitrary,
    title={Arbitrary-Oriented Object Detection with Circular Smooth Label},
    author={Yang, Xue and Yan, Junchi},
    booktitle={European Conference on Computer Vision},
    pages={677--694},
    year={2020}
}
```