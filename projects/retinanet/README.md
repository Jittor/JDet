# RetinaNet

###Preparing
Download pretrained weight to weights/yx_init_pretrained.pk_jt.pk 
###Training
```sh
python run_net.py --config-file=configs/[exp].py --task=train
```
###Testing
```sh
python run_net.py --config-file=configs/[exp].py --task=test
```