import json
import pickle as pk
import jittor as jt
import tensorflow as tf
import numpy as np

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        #return np.ascontiguousarray(v.transpose(3,2,0,1))
        return np.ascontiguousarray(v.transpose(2,3,1,0))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

data_path = "/mnt/disk/cxjyxx_me/JAD/JDet/projects/retinanet/exp/retinanet_11/checkpoints/ckpt_100000.pkl"
pairs = json.load(open("/mnt/disk/cxjyxx_me/JAD/JDet/projects/retinanet/cache_check/init.pk_pairs.json", "r"))
data = jt.load(data_path)["model"]
out_data = {}
for p in pairs:
    out_data[p[0]] = tr(data[p[1]])
pk.dump(out_data, open(data_path+'_tf.pk', "wb"))
print("done")