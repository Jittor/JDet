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
        if (not os.path.exists("test_datas_kld")):
            os.makedirs("test_datas_kld")
        pk.dump(data, open("test_datas_kld/test_data.pk", "wb"))
        print(std_roi_cls_losses)
        print(std_roi_loc_losses)
    else:
        data = pk.load(open("test_datas_kld/test_data.pk", "rb"))
        imagess = jdet.utils.general.to_jt_var(data["imagess"])
        targetss = jdet.utils.general.to_jt_var(data["targetss"])
        std_roi_cls_losses = data["std_roi_cls_losses"]
        std_roi_loc_losses = data["std_roi_loc_losses"]
        # std_roi_cls_losses = [1.1303294 3.1521766 1.1589909 1.2723612 1.223715 1.3785343 1.2342198 1.2582443 1.1898961 1.1536622 1.1907382 1.1355101]
        # std_roi_loc_losses = [3.1484232 1.2802426 2.1646636 2.757801 2.8626733 2.1977746 3.1959727 2.9873595 2.7312007 3.2445354 2.0728052 2.379803
        for batch_idx in range(len(imagess)):
            images = imagess[batch_idx]
            targets = targetss[batch_idx]

            losses = model(images,targets)
            l1 = losses["roi_cls_loss"].data[0]
            s_l1 = std_roi_cls_losses[batch_idx]
            l2 = losses["roi_loc_loss"].data[0]
            s_l2 = std_roi_loc_losses[batch_idx]
            print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2))
            assert(abs(l1 - s_l1) / abs(s_l1) < 1e-1)
            assert(abs(l2 - s_l2) / abs(s_l2) < 1e-1)
            all_loss,losses = parse_losses(losses)
            optimizer.step(all_loss)
            scheduler.step(iter,0,by_epoch=True)
            iter+=1
    print("success!")
if __name__ == "__main__":
    main()