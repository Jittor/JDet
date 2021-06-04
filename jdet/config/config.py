# reference maskrcnn_benchmark default
from collections import OrderedDict
from jdet.utils.general import check_file
import os
import yaml
import copy

__all__ = ["get_cfg","init_cfg","save_cfg","print_cfg"]

class CfgNode(OrderedDict):

    def __getattr__(self, name):
        if name in self:
            return self[name]
        return None

    def __setattr__(self, name, value):
        self[name] = value

    def merge_from_file(self, cfg_file):
        """Load a yaml config file and merge it this CfgNode."""
        
        if check_file(cfg_file,ext=[".yaml"]):
            with open(cfg_file, "r") as f:
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
        """convert cfgNode to dict"""
        now = dict()
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
    print("Loading config from",yaml_file)
    _cfg.merge_from_file(yaml_file)

def get_cfg():
    return _cfg

def save_cfg(save_file):
    with open(save_file,"w") as f:
        f.write(yaml.dump(_cfg.dump()))

def print_cfg():
    data  =  yaml.dump(_cfg.dump())
    # TODO: data keys are not sorted
    print(data) 
