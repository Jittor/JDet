import numpy as np
import jittor as jt
import cv2 

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
