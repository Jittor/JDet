import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
import argparse
import os
import pickle as pk
import numpy as np
import random

def fake_argsort(x, dim=0, descending=False):
    return jt.index(x)[0], x

def fake_argsort2(x, dim=0, descending=False):
    x_ = x.data
    if (descending):
        x__ = -x_
    else:
        x__ = x_
    index_ = np.argsort(x__, axis=dim, kind="stable")
    y_ = x_[index_]
    index = jt.array(index_)
    y = jt.array(y_)
    return index, y

def fake_sort2(x):
    x_ = x.data
    y_ = np.sort(x_, kind="stable")
    y = jt.array(y_)
    return y

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.sort = fake_sort2
    jt.argsort = fake_argsort2
    jt.flags.use_cuda=1
    jt.set_global_seed(223)
    np.random.seed(0)
    random.seed(0)
    init_cfg("configs/ReDet_re50_refpn_1x_dota1_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)
    model.train()

    if (args.set_data):
        os.makedirs("test_datas_redet",exist_ok=True)
        model.save("test_datas_redet/init_pretrained.pk_jt.pk")

        imagess = []
        targetss = []
        loss_list = []

        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print("batch_idx=" + str(batch_idx))
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))

            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            loss_list.append(all_loss.item())
            if (batch_idx > 10):
                break
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "losses" : loss_list
        }            
        pk.dump(data, open("test_datas_redet/test_data.pk", "wb"))
        print(loss_list)
    else:
        model.load("test_datas_redet/init_pretrained.pk_jt.pk")
        data = pk.load(open("test_datas_redet/test_data.pk", "rb"))
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        loss_list = data["losses"]
        thr = 0.2
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            jt.sync_all(True)
            l = all_loss.item()
            c_l = loss_list[batch_idx]
            err_rate = float(abs(c_l - l)/c_l)
            print(f"correct loss is {float(c_l):.4f}, runtime loss is {float(l):.4f}, err rate is {err_rate*100:.2f}%")
            assert err_rate<thr,f"LOSS is not correct, please check it"
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
    print("success!")
if __name__ == "__main__":
    main()