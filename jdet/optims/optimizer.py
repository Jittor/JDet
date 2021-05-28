
from jdet.utils.registry import OPTIMS 

from jittor.optim import SGD 

OPTIMS.register_module(module=SGD)