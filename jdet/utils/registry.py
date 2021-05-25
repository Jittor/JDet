import inspect

class Registry:
    
    def __init__(self,name):
        self.name = name

    def register(self, name, module=None):
        def register_fn(fn):
            assert name not in self,f"{name} is already registered."
            self[name]=fn
            return fn

        if module is not None:
            # used as function call
            return register_fn(module)
        else:
            # used as decorator
            return register_fn

    def __getattr__(self,name):
        if name in self.__dict__:
            return self.__dict__[name]
        return self[name]
            

DATASETS = Registry()
BACKBONES = Registry()
ROI_HEADS = Registry()
LOSSES = Registry()


