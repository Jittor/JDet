from PIL import Image
import numpy as np 
import os

from jdet.utils.registry import DATASETS
from jdet.utils.general import check_dir, to_jt_var
from .transforms import Compose
from tqdm import tqdm

import jittor as jt 
import os
from jittor.dataset import Dataset 
import jdet

def cal_topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k

    Parameters:
        output: jt.Var [N, K]
        target: jt.Var [K]
    """
    with jt.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = jt.equal(pred, target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdims=True)
            res.append(correct_k.sum().item() * (100.0 / batch_size))
        return res

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

    def evaluate(self,results,work_dir,epoch,logger=None, save=True):
        print("Calculating mAP......")
        if save:
            save_path = os.path.join(work_dir,f"detections/val_{epoch}")
            check_dir(save_path)
            jt.save(results,save_path+"/val.pkl")
        sum_top1, sum_top5, count = 0, 0, 0
        num_classes = len(self.classes)
        sum_intersection, sum_output, sum_target = jt.zeros((num_classes)), jt.zeros((num_classes)), jt.zeros((num_classes))
        for img_idx,(result,target) in tqdm(enumerate(results)):
            target = jt.array([target['img_label']])
            result = to_jt_var(result).unsqueeze(0)
            top1, top5 = cal_topk_accuracy(result, target, topk=(1,5))

            output, confidence = jt.argmax(result, dim=1, keepdims=False)
            intersection = output[output == target]
            sum_intersection = jt.scatter(sum_intersection, 0, intersection, jt.array([1]), reduce='add')
            sum_output = jt.scatter(sum_output, 0, output, jt.array([1]), reduce='add')
            sum_target = jt.scatter(sum_target, 0, target, jt.array([1]), reduce='add')

            sum_top1 = sum_top1 + top1
            sum_top5 = sum_top5 + top5
            count = count + 1
        iou_classes = sum_intersection / (sum_output + sum_target - sum_intersection + 1e-10)
        accuracy_class = sum_intersection / (sum_target + 1e-10)
        aps = dict(
            mIoU = jt.mean(iou_classes, 0).item(),
            mAcc = jt.mean(accuracy_class, 0).item(),
            allAcc = sum_intersection.sum() / (sum_target.sum() + 1e-10),
            mtop1 = sum_top1 / count,
            mtop5 = sum_top5 / count,
        )
        return aps
