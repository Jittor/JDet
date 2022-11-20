## GWD

> [Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss](https://arxiv.org/pdf/2101.11952.pdf)

<!-- [ALGORITHM] -->

### Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/gwd.png" width="800"/>
</div>

Boundary discontinuity and its inconsistency to the final detection metric have been the bottleneck for rotating detection regression loss design. In this paper, we propose a novel regression loss based on Gaussian Wasserstein distance as a fundamental approach to solve the problem. Specifically, the rotated bounding box is converted to a 2- D Gaussian distribution, which enables to approximate the indifferentiable rotational IoU induced loss by the Gaussian Wasserstein distance (GWD) which can be learned efficiently by gradient back-propagation. GWD can still be informative for learning even there is no overlapping between two rotating bounding boxes which is often the case for small object detection. Thanks to its three unique properties, GWD can also elegantly solve the boundary discontinuity and square-like problem regardless how the bounding box is defined. Experiments on five datasets using different detectors show the effectiveness of our approach.

### Training
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota.py --task=train
```
### Testing
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota.py --task=test
```
### Performance
|            Models             | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper | Config     | Download   |
|:-----------------------------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------: | :--------: |
|          GWD-R50-FPN          | DOTA1.0| 1024/200| flip|-|  SGD   |   1x    | 68.88   | [arxiv](https://arxiv.org/abs/2101.11952)| [config](projects/gwd/configs/rotated_retinanet_hbb_gwd_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/b261e36e4cf04d6b830a/?dl=1) |

### Citation

```
@inproceedings{yang2021rethinking,
    title={Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss},
    author={Yang, Xue and Yan, Junchi and Qi, Ming and Wang, Wentao and Xiaopeng, Zhang and Qi, Tian},
    booktitle={International Conference on Machine Learning},
    year={2021}
}
```