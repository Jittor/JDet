from jittor.datasets import Dataset 


import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCODataset(Dataset):

    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    def __init__(self,root,anno_file,transforms=None,batch_size=1,num_workers=0,shuffle=False):
        super(COCODataset,self).__init__(batch_size=batch_size,num_workers=num_workers,shuffle=shuffle)
        self.root = root 
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.total_len = len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        file_name = self.coco.loadImgs(id)[0]["file_name"]
        path = os.path.join(self.root, file_name)
        image = Image.open(path).convert("RGB")
        annos = self.coco.loadAnns(self.coco.getAnnIds(id))



        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target



def test():
    COCODataset(root="/mnt/disk-1/lxl/dataset/coco/images/train2017",anno_file="")