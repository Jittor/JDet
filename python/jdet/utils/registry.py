class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self, name=None,module=None):
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules,f"{key} is already registered."
            self._modules[key]=module
            return module

        if module is not None:
            return _register_module(module)

        return _register_module

    def get(self,name):
        assert name in self._modules,f"{name} is not registered."
        return self._modules[name]


def build_from_cfg(cfg,registry,**kwargs):
    if isinstance(cfg,str):
        return registry.get(cfg)(**kwargs)
    elif isinstance(cfg,dict):
        args = cfg.copy()
        args.update(kwargs)
        obj_type = args.pop('type')
        obj_cls = registry.get(obj_type)
        return obj_cls(**args)
    elif isinstance(cfg,list):
        return nn.Sequential([build_from_cfg(c,registry,**kwargs) for c in cfg])
    elif cfg is None:
        return None
    else:
        raise TypeError(f"type {type(cfg)} not support")


DATASETS = Registry()
TRANSFORMS = Registry()
MODELS = Registry()
BACKBONES = Registry()
HEADS = Registry()
LOSSES = Registry()
OPTIMS = Registry()
BRICKS = Registry()
NECKS = Registry()
SCHEDULERS = Registry()
BOXES = Registry()
HOOKS = Registry()

ROI_EXTRACTORS = Registry()
SHARED_HEADS = Registry()
