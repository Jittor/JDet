from jittor import nn 

from jdet.utils.registry import build_from_cfg,BACKBONES

@BACKBONES.register_module()
class Identity(nn.Module):
    def execute(self,x):
        return x 

