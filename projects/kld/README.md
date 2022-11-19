## KLD
> [Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence](https://arxiv.org/pdf/2106.01883.pdf)

<!-- [ALGORITHM] -->
### Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/kld.png" width="800"/>
</div>

Existing rotated object detectors are mostly inherited from the horizontal detection paradigm, as the latter has evolved into a well-developed area. However, these detectors are difficult to perform prominently in high-precision detection due to the limitation of current regression loss design, especially for objects with large aspect ratios. Taking the perspective that horizontal detection is a special case for rotated object detection, in this paper, we are motivated to change the design of rotation regression loss from induction paradigm to deduction methodology, in terms of the relation between rotation and horizontal detection. We show that one essential challenge is how to modulate the coupled parameters in the rotation regression loss, as such the estimated parameters can influence to each other during the dynamic joint optimization, in an adaptive and synergetic way. Specifically, we first convert the rotated bounding box into a 2-D Gaussian distribution, and then calculate the Kullback-Leibler Divergence (KLD) between the Gaussian distributions as the regression loss. By analyzing the gradient of each parameter, we show that KLD (and its derivatives) can dynamically adjust the parameter gradients according to the characteristics of the object. For instance, it will adjust the importance (gradient weight) of the angle parameter according to the aspect ratio. This mechanism can be vital for high-precision detection as a slight angle error would cause a serious accuracy drop for large aspect ratios objects. More importantly, we have proved that KLD is scale invariant. We further show that the KLD loss can be degenerated into the popular $l_{n}$-norm loss for horizontal detection. Experimental results on seven datasets using different detectors show its consistent superiority

### Training
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_kld_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/rotated_retinanet_hbb_kld_r50_fpn_1x_dota.py --task=test
```

### Performance
|   Models    | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper |                             Config                             | Download   |
|:-----------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------------------------------------------------------------:| :--------: |
| KLD-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 69.10   | [arxiv](https://arxiv.org/abs/2106.01883)| [config](configs/rotated_retinanet_hbb_kld_r50_fpn_1x_dota.py) | [model](https://cloud.tsinghua.edu.cn/f/fa7e892f90304af6988b/?dl=1) |

### Citation
```
@article{yang2021learning,
  title={Learning high-precision bounding box for rotated object detection via kullback-leibler divergence},
  author={Yang, Xue and Yang, Xiaojiang and Yang, Jirui and Ming, Qi and Wang, Wentao and Tian, Qi and Yan, Junchi},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```