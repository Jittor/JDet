
class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"] or:
        f = some_registry.foo_module
    '''

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


