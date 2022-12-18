import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import argparse
import os
import pickle as pk
import jdet

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/rotated_retinanet_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()
    if (args.set_data):
        imagess = []
        targetss = []
        correct_loss = []
        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
        for batch_idx,(images,targets) in enumerate(train_dataset):
            if (batch_idx > 10):
                break
            print(batch_idx)
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            correct_loss.append(all_loss.item())
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "correct_loss": correct_loss,
        }
        if (not os.path.exists("test_datas_rotated_retinanet")):
            os.makedirs("test_datas_rotated_retinanet")
        pk.dump(data, open("test_datas_rotated_retinanet/test_data.pk", "wb"))
        print(correct_loss)
    else:
        model.load_parameters(pk.load(open("test_datas_rotated_retinanet/model.pk", "rb")))
        data = pk.load(open("test_datas_rotated_retinanet/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        correct_loss = data["correct_loss"]
        # correct_loss =[1.852632999420166, 2.030822277069092, 1.9102485179901123, 2.9509782791137695, 2.3653626441955566, 
        # 2.2163989543914795, 2.2501344680786133, 2.3585996627807617, 3.020094633102417, 2.5657663345336914, 3.5694150924682617]
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]
            losses = model(images,targets)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            l = all_loss.item()
            c_l = correct_loss[batch_idx]
            err_rate = abs(c_l-l)/min(c_l,l)
            print(f"correct loss is {c_l:.4f}, runtime loss is {l:.4f}, err rate is {err_rate*100:.2f}%")
            assert err_rate<1e-2,"LOSS is not correct, please check it"
        print(f"Loss is correct with err_rate<{1e-2}")
    print("success!")
    
if __name__ == "__main__":
    main()