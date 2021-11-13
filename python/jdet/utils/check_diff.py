import jittor as jt 
import numpy as np

is_diff = False 

def check_diff(w1_,w2_):
    w1 = jt.load(w1_)
    w2 = jt.load(w2_)
    keys = list(w1.keys())
    keys.sort()
    for k in keys:
        if w1[k][1] is None:continue
        v1 = w1[k][1]
        k = k.strip("module.")
        v2 = w2[k][1]
        # if k == "neck.fpn_convs.2.conv.weight":
        #     print(v1)
        #     print("-------------------------")
        #     print(v2)
        #     print("---------------------------")
        #     print(np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12))
        #     break
        abs_err = (np.abs(v1 - v2)).max()
        rel_err = (np.abs(v1 - v2) / (np.maximum(np.abs(v1), np.abs(v2)) + 1e-12)).max()
        our_err = (np.abs(v1 - v2) / (np.abs(v1) + 1e-12)).max()
        print(f'{k:45}{abs_err:10.5f}{rel_err:10.5f}{our_err:20.5f}')

# check_diff("/home/lxl/workspace/s2anet/torch_grad.pkl","jt_grad.pkl")

def check_init(jt_pkl,torch_pkl):
    params1 = jt.load(jt_pkl)
    params2 = jt.load(torch_pkl)
    for k,p1 in params1.items():
        p2 = params2[k]
        p1m = p1.mean()
        p1s = p1.std()
        p2m = p2.mean()
        p2s = p2.std()
        if not (np.abs(p1m-p2m)<1e-3 and np.abs(p1s-p2s)<1e-3):
            print(f"{k}:{p1.shape}--{p2.shape}--{p1m}!={p2m} and {p1s}!={p2s}")

    


def compare_data(data,name):
    if not is_diff:
        return
    def __sync(data):
        if isinstance(data,(tuple,list)):
            return [__sync(d) for d in data]
        elif isinstance(data,jt.Var):
            return data.numpy()
        else:
            assert False

    import pickle 
    import numpy as np
    data = __sync(data)
    gt_data = pickle.load(open(f"/home/lxl/diff/{name}.pkl","rb"))
    if isinstance(data,np.ndarray):
        data = [data]
        gt_data = [gt_data]
    print("-"*10,name,"-"*10)
    for d1,d2  in zip(data,gt_data):
        print(d1.shape,d2.shape)
        diff = np.abs(d1-d2)
        if diff.max()>1e-2:
            index = np.where(diff==diff.max())
            print(index)
            print(d1[diff==diff.max()],d2[diff==diff.max()])
            print(diff.max())
        np.testing.assert_allclose(d1,d2,atol=1e-2,rtol=1e-3)

def load_data(data, name):
    if not is_diff:
        return data
    def __sync(data):
        if isinstance(data,(tuple,list)):
            return [__sync(d) for d in data]
        elif isinstance(data,np.ndarray):
            return jt.array(data)
        else:
            assert False
    
    import pickle 
    data = pickle.load(open(f"/home/lxl/diff/{name}.pkl","rb"))
    return __sync(data)

def compare_and_load(data,name):
    compare_data(data,name)
    return load_data(data,name)
# check_init("jittor_init_weight.pkl","torch_init_weight.pkl")

