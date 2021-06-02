import jittor as jt 
import time 
import warnings
import os 

def sync(data,reduce_mode="mean",to_numpy=True):
    """
        sync data and convert data to numpy
    """
    def _sync(data):
        if isinstance(data,list):
            data =  [_sync(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_sync(d) if isinstance(d,jt.Var) else d for k,d in data.items()}
        elif isinstance(data,jt.Var):
            if jt.in_mpi:
                data = data.mpi_all_reduce(reduce_mode)
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data,(int,float,str)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _sync(data) 

def current_time():
    return time.asctime( time.localtime(time.time()))

def check_file(file):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    return True

def build_file(work_dir,prefix):
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 

def check_interval(step,step_interval):
    if step is None or step_interval is None:
        return False 
    if step % step_interval==0:
        return True 
    return False 

def check_dir(work_dir):
    os.makedirs(work_dir,exist_ok=True)