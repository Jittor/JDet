# JDet Framework
## 1. Config
It has two part: config & constant, config is the global args,please refer to [config.md](config.md) for details. Constant includes some classnames.

## 2. Ops
This section includes some useful ops with CUDA & C++.

There are:
1. rotated box iou:```box_iou_rotated.py```
2. deformable convolution: ```dcn_v1.py```,```dcn_v2.py```
3. roi transformers: ```bbox_transformer.py```
4. feature refine module: ```fr.py```, which is used in r3det
5. nms for polygons,rotated boxes,and horizontal boxes:```nms.py```,```nms_poly.py```,```nms_rotated```
6. ActiveRotatingFilter: ```orn.py```,which is used in s2anet
7. position sensitive roi align: ```psroi_align.py```
8. roi align & roi pooling: ```roi_align.py```,```roi_pool.py```
9. roi align for rotated boxes: ```roi_align_rotated.py```, which is used in gliding

**Notice**:
1. The format of rotated box is [x_center,y_center,width,height,angle].
2. The format of polygons is [x1,y1,x2,y2,x3,y3,x4,y4].
3. The format of horizontal box is [x1,y1,x2,y2].

## 3. data
### dataset
In this module, we provide dataset for COCO format, yolo format,dota format and custom format(which is like mmdet).

In transforms, we provide Flip,Rotate,Resize,etc,for horizontal and oriented boxes, more transformation operators are coming soon.

**Notice**
1. In Transform & Dataset, ```rboxes``` is for rotated box, ```hboxes & bboxes``` is for horizontal boxes, ```polys``` is for polygons.
2. In dataset, ```filter_empty_gt=True``` means that we will remove the image without targets, otherwise, when we meet image without targets, we will randomly choose another image.
3. In dataset, ```balance_category=True``` means that we will balance the number of images for that the number of each category targets is close, by repeating some images. 
### devkits
In this module, we provide the code of preprocess,evaluation and merge devkits. 
Details are shown in [dota.md](dota.md)

## 4. models

## 5. optims
### optimizer

### lr_scheduler

## 6. runner

## 7. utils
In ```general.py```, we 


