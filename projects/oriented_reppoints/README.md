## Oriented RepPoints
> [Oriented RepPoints for Aerial Object Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->
### Abstract

<div align=center>
<img src="https://raw.githubusercontent.com/zytx121/image-host/main/imgs/oriented_reppoints.png" width="800"/>
</div>

In contrast to the generic object, aerial targets are often non-axis aligned with arbitrary orientations having
the cluttered surroundings. Unlike the mainstreamed approaches regressing the bounding box orientations, this paper
proposes an effective adaptive points learning approach to aerial object detection by taking advantage of the adaptive
points representation, which is able to capture the geometric information of the arbitrary-oriented instances.
To this end, three oriented conversion functions are presented to facilitate the classification and localization
with accurate orientation. Moreover, we propose an effective quality assessment and sample assignment scheme for
adaptive points learning toward choosing the representative oriented reppoints samples during training, which is
able to capture the non-axis aligned features from adjacent objects or background noises. A spatial constraint is
introduced to penalize the outlier points for roust adaptive learning. Experimental results on four challenging
aerial datasets including DOTA, HRSC2016, UCAS-AOD and DIOR-R, demonstrate the efficacy of our proposed approach.

### Training
```sh
python run_net.py --config-file=configs/oriented_reppoints_r50_fpn_1x_dota.py --task=train
```

### Testing
```sh
python run_net.py --config-file=configs/oriented_reppoints_r50_fpn_1x_dota.py --task=test
```

### Performance
|   Models    | Dataset| Sub_Image_Size/Overlap |Train Aug | Test Aug | Optim | Lr schd | mAP    | Paper |                             Config                             | Download   |
|:-----------:| :-----: |:-----:|:-----:| :-----: | :-----:| :-----:| :----: |:--------:|:--------------------------------------------------------------:| :--------: |
| Oriented-Reppoints-R50-FPN | DOTA1.0|1024/200| flip|-|  SGD   |   1x    | 66.99   | [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Oriented_RepPoints_for_Aerial_Object_Detection_CVPR_2022_paper.pdf)| [config](configs/oriented_reppoints_r50_fpn_1x_dota.py) | [model]() |

### Citation
```
@inproceedings{li2022ori,
    title={Oriented RepPoints for Aerial Object Detection},
    author={Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu},
    booktitle={Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```