# KFIoU

> [The KFIoU Loss for Rotated Object Detection](https://arxiv.org/abs/2201.12558)

<!-- [ALGORITHM] -->

## Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/kfiou.png" width="800"/>
</div>

Differing from the well-developed horizontal object detection area whereby the computing-friendly IoU based loss is
readily adopted and well fits with the detection metrics. In contrast, rotation detectors often involve a more
complicated loss based on SkewIoU which is unfriendly to gradient-based training. In this paper, we argue that one
effective alternative is to devise an approximate loss who can achieve trend-level alignment with SkewIoU loss instead
of the strict value-level identity. Specifically, we model the objects as Gaussian distribution and adopt Kalman filter to
inherently mimic the mechanism of SkewIoU by its definition, and show its alignment with the SkewIoU at trend-level. This
is in contrast to recent Gaussian modeling based rotation detectors e.g. GWD, KLD that involves a human-specified
distribution distance metric which requires additional hyperparameter tuning. The resulting new loss called KFIoU is
easier to implement and works better compared with exact SkewIoU, thanks to its full differentiability and ability to
handle the non-overlapping cases. We further extend our technique to the 3-D case which also suffers from the same
issues as 2-D detection. Extensive results on various public datasets (2-D/3-D, aerial/text/face images) with different
base detectors show the effectiveness of our approach.

### Training
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota.py --task=test
```

## Performance

|   Models    | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper |                             Config                             | Download   |
|:-----------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------------------------------------------------------------:| :--------: |
| KFIoU-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 69.36   | [arxiv](https://arxiv.org/abs/2201.12558)| [config](configs/rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/9d24118280864511b943/?dl=1) |

## Citation

```
@article{yang2022kfiou,
      title={The KFIoU Loss for Rotated Object Detection},
      author={Xue Yang and Yue Zhou and Gefan Zhang and Jirui Yang and Wentao Wang and Junchi Yan and Xiaopeng Zhang and Qi Tian},
      journal={arXiv preprint arXiv:2201.12558},
      year={2022},
}
```