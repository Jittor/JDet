from jdet.config import init_cfg,get_cfg

init_cfg("configs/s2anet_r50_fpn_1x_dota.py")
cfg = get_cfg()

print(cfg.work_dir)