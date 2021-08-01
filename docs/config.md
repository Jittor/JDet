# How to use configs in JDet
## Basic usages
You need to create a configuration file first, only yaml and py format configuration files are supported.
```yaml
# cfg.yaml
exp_name: resnet
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
```
And you can set this file as a global configuration var by:
```python
# main.py
from jdet.config import init_cfg
init_cfg('cfg.yaml')
```
Then you can get the configuration var in other files:
```python
# model.py
from jdet.config import get_cfg
cfg = get_cfg()
# you can use attributes in cfg by cfg.xxx.xxx
print(cfg.exp_name) # resnet
print(cfg.model.return_stages) # [layer1,layer2,layer3,layer4]
```
You can also print this configuration var or dump this configuration var into a yaml file:
```python
# model.py
from jdet.config import save_cfg, print_cfg
save_cfg('your_output_path.yaml')
print_cfg()
```
## Advanced usages
### .py configuration files
You can do some easy computation in the .py configuration file:
```python
# cfg.py
import os
exp_id = 1
# path setting
output_path = 'experiments'
root_path = os.path.join(output_path, str(exp_id))
log_path = os.path.join(root_path, 'logs')

# easy calculation
gpus = [0,1,2,3]
n_gpus = len(gpus)
batch_size = 16
base_lr = batch_size * 0.001

# model setting
model = {
    'type': 'Resnet50',
    'return_stages': = ['layer1','layer2','layer3','layer4'],
    'pretrained': True
}
```
The function of this file is the same with the following file:
```yaml
# cfg.yaml
exp_id = 1

output_path = 'experiments'
root_path = 'experiments/1'
log_path = 'experiments/1/logs'

gpus = [0,1,2,3]
n_gpus = 4
batch_size = 16
base_lr = 0.016

model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
```
You can load .py configuration file as load .yaml configuration file:
```python
# main.py
from jdet.config import init_cfg
init_cfg('cfg.py')
```
### Inheritance
Some basic configurations may be used by many configuration files, thus we support the inheritance in different configuration files by the `_base_` key:
```yaml
# base.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True

# exp_1.yaml
_base_: base.yaml
name: exp_1
lr: 0.01
batch_size: 1

# exp_1.yaml has the same function as:
# exp_1_no_base.yaml
name: exp_1
lr: 0.01
batch_size: 1
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True

# exp_2.yaml
_base_: base.yaml
name: exp_2
lr: 0.08
batch_size: 8

# exp_2.yaml has the same function as:
# exp_2_no_base.yaml
name: exp_2
lr: 0.08
batch_size: 8
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
```
Multiple base configuration files are also supported:
```yaml
# base1.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True

# base2.yaml
dataset:
    type: COCODataset
    root: /mnt/disk/lxl/dataset/coco/images/train2017
    num_workers: 4
    shuffle: True

# cfg.yaml
_base_: [base1.yaml, base2.yaml]
name: exp
lr: 0.01
```
Note that the priority of configuration attributes is: the final configuration file(cfg.yaml) > the configuration file at the bottom of the `_base_` list(base2.yaml) > the configuration file at the top of the `_base_` list(base1.yaml). This is an example:
```yaml
# base1.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: COCODataset
    root: /mnt/disk/lxl/dataset/coco/images/train2017
    num_workers: 4
    shuffle: True

# base2.yaml
model:
    type: Resnet101

# cfg.yaml
_base_: [base1.yaml, base2.yaml]
dataset:
    num_workers: 8

# cfg.yaml has the same function as:
# cfg_no_base.yaml
model:
    type: Resnet101
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: COCODataset
    root: /mnt/disk/lxl/dataset/coco/images/train2017
    num_workers: 8
    shuffle: True
```

### Cover
Sometimes you may need to cover all parts of an attribute in the base configuration file instead of one of its sub-attributes, thus we provide the key `_cover_`. You can set `_cover_=True` is some attribute, then this attribute will cover all parts of this attribute in the base configuration file:
```yaml
# base.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: COCODataset
    root: /mnt/disk/lxl/dataset/coco/images/train2017
    num_workers: 4
    shuffle: True

# cfg.yaml
_base_: [base.yaml]
dataset:
    _cover_: True
    type: MyDataset

# cfg.yaml has the same function as:
# cfg_no_base.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: MyDataset
```
Please note that `_cover_` is only works in the inheritance relationship, and it will not work between two files inherited by one file at the same time:
```yaml
# base1.yaml
model:
    type: Resnet50
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: COCODataset
    root: /mnt/disk/lxl/dataset/coco/images/train2017
    num_workers: 4
    shuffle: True

# base2.yaml
model:
    _cover_: True   # will not work for base1.yaml
    type: Resnet101

# cfg.yaml
_base_: [base1.yaml, base2.yaml]
dataset:
    _cover_: True   # will work
    type: MyDataset

# cfg.yaml has the same function as:
# cfg_no_base.yaml
model:
    type: Resnet101
    return_stages:  [layer1,layer2,layer3,layer4]
    pretrained: True
dataset:
    type: MyDataset
```
Please note that the keys `_base_` and `_cover_`  are also supported in the .py configuration file.
We are considering whether to support `_cover_` between brother configurations, if you need this feature, please let us know.

Please refer to `[ROOT]/python/jdet/config/config.py` and `[ROOT]/tests/test_config` for more details.