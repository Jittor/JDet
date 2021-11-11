
from PIL import Image
import numpy as np 
import os

from jdet.utils.registry import DATASETS
from .transforms import Compose

import jittor as jt 
import os
from jittor.dataset import Dataset 
import jdet

@DATASETS.register_module()
class ImageDataset(Dataset):
    """ ImageDataset
    Load image without groundtruth for visual or test
    """
    def __init__(self,images_file=None,
                      images_dir="",
                      dataset_type="DOTA",
                      transforms=[
                          dict(
                              type="Resize",
                              min_size=[800],
                              max_size=1333
                          ),
                          dict(
                              type="Pad",
                              size_divisor=32
                          ),
                          dict(
                              type="Normalize",
                              mean=[123.675, 116.28, 103.53],
                              std = [58.395, 57.12, 57.375],
                          )
                      ],
                      batch_size=1,
                      num_workers=0,
                      shuffle=False):
        super(ImageDataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle)
        self.images_file = self._load_images(images_file,images_dir=images_dir)
        self.total_len = len(self.images_file)
        self.dataset_type = dataset_type

        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be list or callable")
        self.transforms = transforms
    
    def _load_images(self,images_file,images_dir):
        if (not images_file):
            images = []
            for name in os.listdir(images_dir):  
                if (jdet.utils.general.is_img(name)):
                    images.append(name)
        elif isinstance(images_file,list):
            pass 
        elif isinstance(images_file,str):
            if os.path.exists(images_file):
                images_file_ = jt.load(images_file)
                images = []
                for i in images_file_:
                    if (isinstance(i, dict)):
                        images.append(i["filename"])
                    elif (isinstance(i, str)):
                        images.append(i)
                    else:
                        raise NotImplementedError
            else:
                assert False,f"{images_file} must be a file or list" 
        else:
            raise NotImplementedError
        
        images = [os.path.join(images_dir, i) for i in images]
        return images

    def __getitem__(self,index):
        if "BATCH_IDX" in os.environ:
            index = int(os.environ['BATCH_IDX'])

        img = Image.open(self.images_file[index]).convert("RGB")
        targets = dict(
            ori_img_size=img.size,
            img_size=img.size,
            scale_factor=1.,
            img_file = self.images_file[index]
        )

        if self.transforms:
            img,targets = self.transforms(img,targets)
        return img,targets 
    
    def collate_batch(self,batch):
        imgs = []
        anns = []
        max_width = 0
        max_height = 0
        for image,ann in batch:
            height,width = image.shape[-2],image.shape[-1]
            max_width = max(max_width,width)
            max_height = max(max_height,height)
            imgs.append(image)
            anns.append(ann)
        N = len(imgs)
        batch_imgs = np.zeros((N,3,max_height,max_width),dtype=np.float32)
        for i,image in enumerate(imgs):
            batch_imgs[i,:,:image.shape[-2],:image.shape[-1]] = image
        
        return batch_imgs,anns 
        