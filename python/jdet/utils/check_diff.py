import torch
import jittor as jt 
import numpy as np

def check_diff(w1_,w2_):
    w1 = jt.load(w1_)
    w2 = jt.load(w2_)
    keys = list(w1.keys())
    keys.sort()
    for k in keys:
        if w1[k][1] is None:continue
        v1 = w1[k][1]
        k = k.strip("module.")
        v2 = w2[k][1]
        # if k == "neck.fpn_convs.2.conv.weight":
        #     print(v1)
        #     print("-------------------------")
        #     print(v2)
        #     print("---------------------------")
        #     print(np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12))
        #     break
        abs_err = (np.abs(v1 - v2)).max()
        rel_err = (np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12)).max()
        our_err = (np.abs(v1 - v2) / (np.abs(v1) + 1e-12)).max()
        print(f'{k:45}{abs_err:10.5f}{rel_err:10.5f}{our_err:20.5f}')

check_diff("/home/lxl/workspace/s2anet/torch_grad.pkl","jt_grad.pkl")
