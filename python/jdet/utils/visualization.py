import numpy as np
import jittor as jt
import cv2 
import os
from jdet.config import COCO_CLASSES

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

def draw_box(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 

def draw_rbox(img,box,text,color):
    box = [int(x) for x in box]
    img = cv2.rectangle(img=img, pt1=tuple(box[0:2]), pt2=tuple(box[2:]), color=color, thickness=1)
    img = cv2.putText(img=img, text=text, org=(box[0],box[1]-5), fontFace=0, fontScale=0.5, color=color, thickness=1)
    return img 

def draw_mask(img,box,mask,text,color):
    pass

def draw_boxes(img,boxes,cats):
    if isinstance(img,jt.Var):
        img = img.numpy()
    for box,cat in zip(boxes,cats):
        img = draw_box(img,box,cat,(255,0,0))
    cv2.imwrite("test.png",img)

def visualize_results(detections,classes,files,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for (bboxes,scores,labels),img_f in zip(detections,files):
        if hasattr(bboxes,"numpy"):
            bboxes = bboxes.numpy()
        if hasattr(scores,"numpy"):
            scores = scores.numpy()
        if hasattr(labels,"numpy"):
            labels = labels.numpy()
        cats = [classes[l-1] for l in labels]
        img = cv2.imread(img_f)
        print(len(cats))
        for box,cate,score in zip(bboxes,cats,scores):
            text = f"{cate}:{score:.2f}"
            img = draw_box(img,box,text,(255,0,0))
        cv2.imwrite(os.path.join(save_dir,img_f.split("/")[-1]),img)
    
def visual_gts(targets,save_dir):
    for t in targets:
        bbox = t["bboxes"]
        labels = t["labels"]
        classes = t["classes"]
        ori_img_size = t["ori_img_size"]
        img_size = t["img_size"]
        bbox[:,0::2] *= (ori_img_size[0]/img_size[0])
        bbox[:,1::2] *= (ori_img_size[1]/img_size[1])
        img_f = t["img_file"]
        img = cv2.imread(img_f)
        for box,l in zip(bbox,labels):
            text = classes[l-1]
            img = draw_box(img,box,text,(255,0,0))
        cv2.imwrite(os.path.join(save_dir,"test.jpg"),img)
            
        

