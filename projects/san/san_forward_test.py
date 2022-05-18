import jittor as jt
jt.set_global_seed(0)
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import numpy as np
import random
import jdet
import argparse
import os
import pickle as pk

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(0)
    np.random.seed(0)
    random.seed(0)
    init_cfg("configs/san10_pairwise.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    numpy_save_dir = 'test_datas_san/models_numpy.pkl'
    numpy_dict = pk.load(open(numpy_save_dir, 'rb'))
    jittor_dict = dict()
    for k, v in numpy_dict.items():
        jittor_dict[k] = jt.array(v)
    model.load_state_dict(jittor_dict)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=model.parameters())

    model.eval()
    if (args.set_data):

        imagess = []
        targetss = []
        correct_loss = []

        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print(batch_idx)
            if (batch_idx > 10):
                break
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            losses = model(images,targets)
            all_loss, losses = parse_losses(losses)
            jt.sync_all(True)
            correct_loss.append(all_loss.item())
            optimizer.step(all_loss)
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        pk.dump(data, open("test_datas_san/test_data_jittor.pkl", "wb"))
        print(correct_loss)
        correct_loss = [jdet.utils.general.sync(i) for i in correct_loss]
        data_numpy = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        pk.dump(data_numpy, open("test_datas_san/test_data_numpy.pkl", "wb"))
    else:
        data = pk.load(open("test_datas_san/test_data_numpy.pkl", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        correct_loss = data["correct_loss"]
        thr = 0.5           #TODO: fix thr
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            all_loss, losses = parse_losses(losses)
            jt.sync_all(True)
            l = all_loss.item()
            # print(l)
            # optimizer.step(all_loss)
            c_l = correct_loss[batch_idx]
            err_rate = float(abs(c_l - l)/np.minimum(c_l, l))
            print(f"correct loss is {float(c_l):.4f}, runtime loss is {float(l):.4f}, err rate is {err_rate*100:.2f}%")
            assert err_rate<thr,f"LOSS is not correct, please check it"
        #print(f"Loss is correct with err_rate<{thr}")
    print("success!")
    
if __name__ == "__main__":
    main()