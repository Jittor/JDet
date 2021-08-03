import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.general import parse_losses
from jdet.utils.registry import build_from_cfg,MODELS,DATASETS,OPTIMS
import numpy as np
import random

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(0)
    np.random.seed(0)
    random.seed(0)
    init_cfg("configs/faster_rcnn_obb_r50_fpn_1x_dota_test.py")
    cfg = get_cfg()

    model = build_from_cfg(cfg.model,MODELS)
    train_dataset = build_from_cfg(cfg.dataset.train,DATASETS)
    optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params= model.parameters())

    model.train()
    correct_loss =[[0.6923643946647644, 0.013670142740011215, 2.6186561584472656, 99.51171875, 0.01625995896756649, 3.3409507274627686],
    [0.6921592950820923, 0.09989452362060547, 2.589883327484131, 96.58203125, 0.14952516555786133, 3.5314621925354004],
    [0.6940275430679321, 0.13187119364738464, 2.5293242931365967, 94.04296875, 0.4167262315750122, 3.771949291229248],
    [0.6803557872772217, 0.17590658366680145, 2.4508888721466064, 90.0390625, 0.148127943277359, 3.4552793502807617],
    [0.6979867219924927, 0.2551208436489105, 2.3485074043273926, 79.39453125, 0.8913882970809937, 4.193003177642822],
    [0.6433444023132324, 0.03799542412161827, 2.0324559211730957, 99.31640625, 0.04919293895363808, 2.762988567352295],
    [0.6496917605400085, 0.2240673154592514, 1.3327971696853638, 93.45703125, 0.3448884189128876, 2.5514447689056396],
    [0.5190461874008179, 0.046668440103530884, 0.49859869480133057, 94.04296875, 0.3457503616809845, 1.410063624382019],
    [0.4358431398868561, 0.01077353022992611, 0.08509550988674164, 99.51171875, 0.012026340700685978, 0.5437385439872742],
    [0.7473233342170715, 0.4877190887928009, 2.8384358882904053, 83.69140625, 1.3975961208343506, 5.471074104309082],
    [0.6330549120903015, 0.10072168707847595, 0.8418699502944946, 90.13671875, 0.33757248520851135, 1.9132189750671387]]
    thr = 0.1
    for batch_idx,(images,targets) in enumerate(train_dataset):
        losses = model(images,targets)
        if (batch_idx > 10):
            break
        all_loss, losses = parse_losses(losses)
        optimizer.step(all_loss)
        l = [losses['loss_rpn_cls'].item(), losses['loss_rpn_bbox'].item(), losses['rbbox_loss_cls'].item(), losses['rbbox_acc'].item(), losses['rbbox_loss_bbox'].item(), all_loss.item()]
        c_l = correct_loss[batch_idx]
        err_rate = float(abs(c_l[-1] - l[-1])/np.minimum(c_l[-1], l[-1]))
        print(f"correct loss is {float(c_l[-1]):.4f}, runtime loss is {float(l[-1]):.4f}, err rate is {err_rate*100:.2f}%")
        assert err_rate<thr,f"LOSS is not correct, please check it"
    print(f"Loss is correct with err_rate<{thr}")
    
if __name__ == "__main__":
    main()