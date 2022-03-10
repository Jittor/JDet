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
class ILSVRCDataset(Dataset):
    """ ILSVRCDataset
    Load image for ILSVRC2012.
    prepare data as format below:

    images_dir/label1/img1.png
    images_dir/label1/img2.png
    ...
    images_dir/label2/img1.png
    images_dir/label2/img2.png
    """
    def __init__(self,images_dir=None,transforms=None,batch_size=1,num_workers=0,shuffle=False,drop_last=False):
        super(ILSVRCDataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,drop_last=drop_last)
        self.classes, self.class_to_idx = self._load_labels(images_dir=images_dir)
        self.images, self.labels = self._load_images(images_dir=images_dir)
        self.total_len = len(self.labels)

        if isinstance(transforms,list):
            transforms = Compose(transforms)
        if transforms is not None and not callable(transforms):
            raise TypeError("transforms must be list or callable")
        self.transforms = transforms
    
    def _load_labels(self, images_dir):
        classes = sorted([d.name for d in os.scandir(images_dir) if d.is_dir()])
        class_to_idx = {v:k for k,v in enumerate(classes)}
        return classes, class_to_idx
 
    def _load_images(self, images_dir):
        images, labels = [], []
        for label in os.listdir(images_dir):
            label_dir = os.path.join(images_dir, label)
            if os.path.isdir(label_dir):
                if label not in self.class_to_idx.keys():
                    raise ValueError("unknow class {}".format(label))
                for name in os.listdir(label_dir):
                    if (jdet.utils.general.is_img(name)):
                        images.append(os.path.join(images_dir, label, name))
                        labels.append(self.class_to_idx[label])
        return images, labels
    
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
        

    def __getitem__(self,index):
        if "BATCH_IDX" in os.environ:
            index = int(os.environ['BATCH_IDX'])

        img = Image.open(self.images[index]).convert("RGB")
        targets = dict(
            ori_img_size=img.size,
            img_size=img.size,
            scale_factor=1.,
            img_file = self.images[index],
            img_label = self.labels[index]
        )

        if self.transforms:
            img,targets = self.transforms(img,targets)
        return img,targets 
