import pickle as pk
import json as js
import numpy as np
import tensorflow as tf
import jittor as jt
import os

def _show_keys():
    data = pk.load(open("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk", "rb"))
    ks = list(data.keys())
    ks.sort()
    for k in ks:
        # if (k.startswith("resnet50_v1d/C1")):
        print(k)

def get_var(name):
    path = 'share_outputs/'+name+'.ckpt'
    if (os.path.exists(path)):
        reader = tf.train.NewCheckpointReader(path)
        var = reader.get_tensor(name)  #numpy.ndarray
    else:
        path = 'share_outputs/'+name+'.pk'
        var = pk.load(open(path, "rb"))  #numpy.ndarray
    x = jt.array(var)
    if (len(x.shape) == 4):
        x = x.permute([0,3,1,2])
    return x