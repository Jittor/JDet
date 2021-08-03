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
    correct_loss =[[0.6910267472267151, 0.015082551166415215, 2.7885429859161377, 0.1953125, 0.05786817893385887, 3.552520513534546],
    [0.6971002221107483, 0.09298689663410187, 2.7481327056884766, 10.7421875, 0.12133262306451797, 3.659552574157715],
    [0.6805869340896606, 0.13677766919136047, 2.6452531814575195, 83.49609375, 0.5078635215759277, 3.9704813957214355],
    [0.6748155355453491, 0.19104263186454773, 2.4975414276123047, 89.74609375, 0.1684001386165619, 3.531799554824829],
    [0.7020979523658752, 0.2751629054546356, 2.1917152404785156, 76.953125, 1.0208370685577393, 4.189813137054443],
    [0.6526368856430054, 0.04270350933074951, 0.7479208111763, 99.4140625, 0.040559880435466766, 1.4838210344314575],
    [0.6510735154151917, 0.3285956382751465, 2.2181060314178467, 90.8203125, 0.546872615814209, 3.744647741317749],
    [0.5742478370666504, 0.03992677107453346, 0.4423956573009491, 97.8515625, 0.1069965660572052, 1.1635668277740479],
    [0.5168476700782776, 0.009665747173130512, 0.3958197236061096, 98.6328125, 0.03502177819609642, 0.9573549032211304],
    [0.7065381407737732, 0.9011915326118469, 0.8094715476036072, 89.94140625, 0.4663526713848114, 2.883553981781006],
    [0.6244182586669922, 0.15812541544437408, 0.7723376750946045, 89.84375, 0.35137325525283813, 1.906254529953003]]
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