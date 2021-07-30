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

# check_diff("/home/lxl/workspace/s2anet/torch_grad.pkl","jt_grad.pkl")

def check_init(jt_pkl,torch_pkl):
    params1 = jt.load(jt_pkl)
    params2 = jt.load(torch_pkl)
    for k,p1 in params1.items():
        p2 = params2[k]
        p1m = p1.mean()
        p1s = p1.std()
        p2m = p2.mean()
        p2s = p2.std()
        if not (np.abs(p1m-p2m)<1e-3 and np.abs(p1s-p2s)<1e-3):
            print(f"{k}:{p1.shape}--{p2.shape}--{p1m}!={p2m} and {p1s}!={p2s}")

    

# check_init("jittor_init_weight.pkl","torch_init_weight.pkl")

