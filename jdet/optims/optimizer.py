
from jdet.utils.registry import OPTIMS 

from jittor import optim 

@OPTIMS.register_module()
class SGD(optim.SGD):
    def state_dict(self):
        data = {}
        for k,d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k]=d 
        return data
