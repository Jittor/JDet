# reference maskrcnn_benchmark default
from collections import OrderedDict
import os
import yaml
import copy

__all__ = ["get_cfg","init_cfg","write_cfg"]

class CfgNode(OrderedDict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return None

    def __setattr__(self, name, value):
        self[name] = value

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it this CfgNode."""

        if not os.path.exists(cfg_filename):
            raise AttributeError(f"{cfg_filename} not exists")

        if not os.path.splitext(cfg_filename)[1] in [".yaml"]:
            raise AttributeError("only support yaml file")


        with open(cfg_filename, "r") as f:
            cfg = yaml.safe_load(f.read())
        self.update(self.dfs(cfg))
    
    def dfs(self,cfg_other):
        if isinstance(cfg_other,dict):
            now_cfg = CfgNode()
            for k,d in cfg_other.items():
                now_cfg[k]=self.dfs(d)
        elif isinstance(cfg_other,list):
            now_cfg = [self.dfs(d) for d in cfg_other]
        else:
            now_cfg = copy.deepcopy(cfg_other)
        return now_cfg
    
    def dump(self):
        now = {}
        for k,d in self.items():
            if isinstance(d,CfgNode):
                d = d.dump()
            if isinstance(d,list):
                d = [dd.dump() if isinstance(dd,CfgNode) else dd for dd in d]
            now[k]=d
        return now
        

_cfg = CfgNode()
_cfg.merge_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),"default.yaml"))

def init_cfg(yaml_file):
    global _cfg
    _cfg.merge_from_file(yaml_file)

def get_cfg():
    global _cfg
    return _cfg

def write_cfg(save_file):
    global _cfg 
    with open(save_file,"w") as f:
        f.write(yaml.dump(_cfg.dump()))
