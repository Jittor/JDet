import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
import models
import argparse
import os
import pickle as pk

def main():
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--set_data",
        action='store_true'
    )
    args = parser.parse_args()

    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/retinanet_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    model.load(cfg.pretrained_weights)
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)
    model.train()

    if (args.set_data):
        imagess = []
        targetss = []
        std_roi_cls_losses = []
        std_roi_loc_losses = []

        train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
        for batch_idx,(images,targets) in enumerate(train_dataset):
            print(batch_idx)
            imagess.append(jdet.utils.general.sync(images))
            targetss.append(jdet.utils.general.sync(targets))
            # print(targets)
            # print(jdet.utils.general.sync(targets))

            losses = model(images,targets)
            l1 = losses["roi_cls_loss"].data[0]
            std_roi_cls_losses.append(l1)
            l2 = losses["roi_loc_loss"].data[0]
            std_roi_loc_losses.append(l2)
            if (batch_idx > 10):
                break
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
        data = {
            "imagess": imagess,
            "targetss": targetss,
            "std_roi_cls_losses": std_roi_cls_losses,
            "std_roi_loc_losses": std_roi_loc_losses
        }
        if (not os.path.exists("test_datas_retinanet")):
            os.makedirs("test_datas_retinanet")
        pk.dump(data, open("test_datas_retinanet/test_data.pk", "wb"))
        print(std_roi_cls_losses)
        print(std_roi_loc_losses)
    else:
        data = pk.load(open("test_datas_retinanet/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        std_roi_cls_losses = data["std_roi_cls_losses"]
        std_roi_loc_losses = data["std_roi_loc_losses"]
        # std_roi_cls_losses = [1.1452212, 1.1484368, 1.1538603, 1.1621443, 1.1542724, 1.1430459, 1.1834915, 1.1830766, 1.5154903, 1.1654731, 1.1685958, 1.1577367]
        # std_roi_loc_losses = [0.17455772, 0.3866686, 0.2991456, 0.232427, 0.2683379, 0.33200195, 0.39404622, 0.39015254, 0.23770154, 0.3625164, 0.3487473, 0.3281753]
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            l1 = losses["roi_cls_loss"].data[0]
            s_l1 = std_roi_cls_losses[batch_idx]
            l2 = losses["roi_loc_loss"].data[0]
            s_l2 = std_roi_loc_losses[batch_idx]
            print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2))
            assert(abs(l1 - s_l1) / abs(s_l1) < 1e-3)
            assert(abs(l2 - s_l2) / abs(s_l2) < 1e-3)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
    print("success!")
if __name__ == "__main__":
    main()