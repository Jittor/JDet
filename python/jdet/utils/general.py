import jittor as jt 
import time 
import warnings
import os 
from functools import partial
from six.moves import map, zip

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

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def current_time():
    return time.asctime( time.localtime(time.time()))

def check_file(file,ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
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


def list_files(file_dir):
    if os.path.isfile(file_dir):
        return [file_dir]

    filenames = []
    for f in os.listdir(file_dir):
        ff = os.path.join(file_dir, f)
        if os.path.isfile(ff):
            filenames.append(ff)
        elif os.path.isdir(ff):
            filenames.extend(list_files(ff))

    return filenames


def is_img(f):
    ext = os.splitext(f)[1]
    return ext.lower() in [".jpg",".bmp",".jpeg",".png","tiff"]

def list_images(img_dir):
    img_files = []
    for img_d in img_dir.split(","):
        if len(img_d)==0:
            continue
        if not os.path.exists(img_d):
            raise f"{img_d} not exists"
        img_d = os.path.abspath(img_d)
        img_files.extend([f for f in list_files(img_d) if is_img(f)])
    return img_files