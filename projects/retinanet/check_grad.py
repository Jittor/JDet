import os
import pickle as pk
import numpy as np

reload_weights = False
# reload_weights = True

run_ours = False
# run_ours = True

def check_diff(w1_, w2_):
    w1 = pk.load(open(w1_, "rb"))['model']
    w2 = pk.load(open(w2_, "rb"))
    keys = list(w1.keys())
    keys.sort()
    for k in keys:
        v1 = w1[k]
        v2 = w2[k]
        abs_err = (np.abs(v1 - v2)).mean()
        rel_err = (np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12)).mean()
        our_err = (np.abs(v1 - v2) / (np.abs(v1) + 1e-12)).mean()
        print(f'{k:45}{abs_err:10.5f}{rel_err:10.5f}{our_err:20.5f}')

def main():
    model = "retinanet_12"
    if (reload_weights):
        weights_path = '/mnt/disk/cxjyxx_me/JAD/RotationDetection/output/trained_weights/RetinaNet_DOTA_2x_20200915'
        init_name = 'DOTA_initmodel.ckpt'
        one_iter_name = 'DOTA_0model.ckpt'

        os.system(f"python convert_weights_tf2jittor.py {os.path.join(weights_path, init_name)} cache_check/init.pk")
        os.system(f"python weights_cvt.py cache_check/init.pk")

        os.system(f"python convert_weights_tf2jittor.py {os.path.join(weights_path, one_iter_name)} cache_check/i1.pk")
        os.system(f"python weights_cvt.py cache_check/i1.pk")
    if (run_ours):
        os.system(f"gopt_disable=1 CUDA_VISIBLE_DEVICES=2 python run_net.py --config-file=configs/{model}.py --use_cuda --task=train")

    ours_weights_path = f'/mnt/disk/cxjyxx_me/JAD/JDet/projects/retinanet/exp/{model}/checkpoints/ckpt_1.pkl'
    yx_weights_path = 'cache_check/i1.pk_jt.pk'
    check_diff(ours_weights_path, yx_weights_path)

    # ours_weights_path = f'/mnt/disk/cxjyxx_me/JAD/JDet/projects/retinanet/exp/{model}/checkpoints/ckpt_0.pkl'
    # yx_weights_path = 'cache_check/init.pk_jt.pk'
    # check_diff(ours_weights_path, yx_weights_path)

main()