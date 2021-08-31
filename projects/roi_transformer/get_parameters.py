import jdet
import jittor as jt
from jdet.config import init_cfg, get_cfg
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.runner import Runner

def main():
    jt.flags.use_cuda=1
    jt.set_global_seed(223)

    init_cfg("configs/faster_rcnn_RoITrans_r50_fpn_1x_dota.py")
    cfg = get_cfg()
    runner = Runner()
    runner.model.save("weights/init_pretrained.pk_jt.pk")

if __name__ == "__main__":
    main()