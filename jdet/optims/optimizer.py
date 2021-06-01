
from jdet.utils.registry import OPTIMS 

from jittor import optim 


class Optimizer(object):
    def parameters(self):
        data = {}
        for k,d in self.__dict__.items():
            if k == "param_groups":
                continue
            data[k]=d 
        return data

    def load_parameters(self,data):
        if isinstance(data,dict):
            for k,d in data.items():
                if k in self.__dict__:
                    self.__dict__[k]=d 
    
@OPTIMS.register_module()
class SGD(optim.SGD,Optimizer):
    pass 

@OPTIMS.register_module()
class Adam(optim.Adam,Optimizer):
    pass 

