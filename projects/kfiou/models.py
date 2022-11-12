import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS
from jittor import init, Module, Var

@MODELS.register_module()
def NormalPrameterGroupsGenerator(named_params):
    params = []
    for p in named_params:
        params.append(p[1])
    return params

@MODELS.register_module()
def YangXuePrameterGroupsGenerator(named_params, model, conv_bias_grad_muyilpy=1., conv_bias_weight_decay=-1, freeze_prefix=[]):

    def get_model_by_name(name):
        v = model
        key_ = name.split('.')
        end = 0
        for k in key_:
            if isinstance(v, nn.Sequential):
                if (k in v.layers):
                    v = v[k]
                elif k.isdigit() and (int(k) in v.layers):
                    v = v[int(k)]
                else:
                    end=1
                    break
            else:
                if hasattr(v, k):
                    v = getattr(v, k)
                    assert isinstance(v, (Module, Var)), \
                        f"expect a jittor Module or Var, but got <{v.__class__.__name__}>, key: {key}"
                else:
                    end = 1
                    break
        assert(end == 0)
        return v

    normal_group = {
        "params":[],
        "grad_mutilpy":1
    }
    conv_bias_group = {
        "params":[],
        "grad_mutilpy":conv_bias_grad_muyilpy
    }
    if (conv_bias_weight_decay >= 0):
        conv_bias_group["weight_decay"] = conv_bias_weight_decay
    for p in named_params:
        name, param = p
        names = name.split(".")
        m = get_model_by_name(".".join(names[:-1]))
        freeze = False
        for prefix in freeze_prefix:
            if (name.startswith(prefix)):
                freeze = True
                break
        if freeze:
            continue

        if ((isinstance(m, jt.nn.Conv)) and (names[-1] == "bias")):
            conv_bias_group['params'].append(param)
            continue
        normal_group['params'].append(param)
    return [normal_group, conv_bias_group]