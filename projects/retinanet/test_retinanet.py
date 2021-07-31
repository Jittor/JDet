import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
import models

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(666)
    init_cfg("configs/retinanet_test.py")
    cfg = get_cfg()

    iter = 0
    model = build_from_cfg(cfg.model,MODELS)
    model.load(cfg.pretrained_weights)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi) if cfg.dataset.train else None
    if (cfg.parameter_groups_generator):
        params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=model.named_parameters(), model=model)
    else:
        params = model.parameters()
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
    scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=optimizer)

    std_roi_cls_losses = [1.1876216,1.2137866,1.1632628,1.156801,1.21125,1.1542095,1.1644664,1.1560789,1.201098,1.1427929,1.1542232,1.1491848]
    std_roi_loc_losses = [0.27317086,0.4552248,0.29419816,0.33686405,0.37411553,0.5635785,0.31424987,0.21355802,0.5266197,0.32028162,0.4911972,0.23555271]
    model.train()
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        l1 = losses["roi_cls_loss"].data[0]
        s_l1 = std_roi_cls_losses[batch_idx]
        l2 = losses["roi_loc_loss"].data[0]
        s_l2 = std_roi_loc_losses[batch_idx]
        assert(abs(l1 - s_l1) / abs(s_l1) < 1e-3)
        assert(abs(l2 - s_l2) / abs(s_l2) < 1e-3)
        print(abs(l1 - s_l1) / abs(s_l1), abs(l2 - s_l2) / abs(s_l2))
        if (batch_idx > 10):
            break
        all_loss,losses = parse_losses(losses)
        optimizer.step(all_loss)
        scheduler.step(iter,0,by_epoch=True)
        iter+=1

if __name__ == "__main__":
    main()