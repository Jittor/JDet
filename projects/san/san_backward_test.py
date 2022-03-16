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

def get_deep_attr(obj, name):
    sls = name.split('.')
    m = obj
    for s in sls:
        if s in ['0', '1', '2', '3', '4', '5']:
            m = m[int(s)]
        elif hasattr(m, s):
                m = getattr(m, s)
        else:
            m = None
            break
    return m


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
    init_cfg("/home/flowey/remote2/JDet/configs/san10_pairwise.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    numpy_save_dir = '/home/flowey/remote2/JDet/projects/san/test_datas_san/models_numpy.pkl'
    numpy_dict = pk.load(open(numpy_save_dir, 'rb'))
    jittor_dict = dict()
    for k, v in numpy_dict.items():
        jittor_dict[k] = jt.array(v)
    model.load_state_dict(jittor_dict)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=model.parameters())

    model.train()
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
            loss = losses['loss']
            grads = pk.load(open('test_datas_san/data_grad.pkl', 'rb'))
            loss_value = grads.pop('loss_value')
            print(loss.numpy() - loss_value)
            max_diff, max_name = .0, 'None'
            for name, value in grads.items():
                w = get_deep_attr(model, name)
                g = jt.grad(loss, w).numpy()
                d = np.max(np.abs((g - value)))/np.max(np.abs(value))
                print(name, d)
                if d > max_diff:
                    max_diff = d
                    max_name = name
            print(max_name, max_diff)
                    
            return
            all_loss, losses = parse_losses(losses)
            jt.sync_all(True)
            l = all_loss.item()
            optimizer.step(all_loss)
            state_dict2_path = '/home/flowey/remote2/JDet/projects/san/test_datas_san/models_numpy2.pth'
            torch_dict = pk.load(open(state_dict2_path, 'rb'))
            for k, v in model.state_dict().items():
                v = v.numpy()
                print(k, np.max(np.abs(v - torch_dict[k])))
            return
        #print(f"Loss is correct with err_rate<{thr}")
    print("success!")
    
if __name__ == "__main__":
    main()