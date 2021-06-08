from jdet.config import JConfig, _init
_init()

#--mode train/test/demo
#--gpus 1,2,3...
#--resume

cfg1 = JConfig("configs/default.yaml", "configs/1.yaml")
cfg2 = JConfig("configs/default.yaml", "configs/2.yaml")
import model
cfg1.set_as_global()
model.temp()
cfg2.set_as_global()
model.temp()
