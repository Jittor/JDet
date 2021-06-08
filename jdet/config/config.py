from yacs.config import CfgNode as CN
import os

def _init():
    global _global_dict
    _global_dict = {}

def get_cfg():
    return _global_dict['cfg']

class JConfig:
    def __init__(self, default_path, config_path):
        jdet_default_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "default.yaml")
        self.cfg = CN.load_cfg(open(jdet_default_path))
        self.cfg.set_new_allowed(True)
        self.cfg.merge_from_file(default_path)
        self.cfg.set_new_allowed(False)
        self.cfg.merge_from_file(config_path)

    def set_as_global(self):
        _global_dict['cfg'] = self.cfg