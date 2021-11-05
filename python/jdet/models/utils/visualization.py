from jdet.models.boxes.box_ops import rotated_box_to_poly_single
import numpy as np
import jittor as jt
import cv2 
import os
from jdet.config import COCO_CLASSES
from jdet.models.boxes.box_ops import rotated_box_to_poly_single

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

def visualize_r_result(img,detection,out_path='test.jpg'):
    bboxes = detection[0][:, :5]
    scores = detection[0][:, 5]
    labels = detection[1]
    if hasattr(bboxes,"numpy"):
        bboxes = bboxes.numpy()
    if hasattr(scores,"numpy"):
        scores = scores.numpy()
    if hasattr(labels,"numpy"):
        labels = labels.numpy()
    print(bboxes.shape)
    for box,label,score in zip(bboxes,labels,scores):
        box = rotated_box_to_poly_single(box)
        box = box.reshape(-1,2).astype(int)
        text = f"{label}:{score:.2}"
        draw_poly(img,box,color=(255,0,0),thickness=2)

        img = cv2.putText(img=img, text=text, org=(box[0][0],box[0][1]-5), fontFace=0, fontScale=0.5, color=(255,0,0), thickness=1)
        
    cv2.imwrite(out_path,img)

def visualize_r_result_boxes(img,detection,out_path='test.jpg'):
    bboxes = detection[:, :5]
    if hasattr(bboxes,"numpy"):
        bboxes = bboxes.numpy()
    for i in range(bboxes.shape[0]):
        box = bboxes[i]
        box = rotated_box_to_poly_single(box)
        box = box.reshape(-1,2).astype(int)
        draw_poly(img,box,color=(255,0,0),thickness=2)
    cv2.imwrite(out_path,img)

def draw_poly(img,point,color,thickness):
    cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)

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
            

def draw_poly(img,point,color,thickness):
    cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)

def draw_rboxes(img_file,boxes,scores,labels,classnames):
    img = cv2.imread(img_file)
    for box,score,label in zip(boxes,scores,labels):
        box = rotated_box_to_poly_single(box)
        box = box.reshape(-1,2).astype(int)
        classname = classnames[label-1]
        text = f"{classname}:{score:.2}"
        draw_poly(img,box,color=(255,0,0),thickness=2)

        img = cv2.putText(img=img, text=text, org=(box[0][0],box[0][1]-5), fontFace=0, fontScale=0.5, color=(255,0,0), thickness=1)

    cv2.imwrite("/mnt/disk/czh/gliding/visualization/test.jpg",img)


def test():
    img_file = "/mnt/disk/lxl/dataset/DOTA_1024/trainval_split/images/P1305__1.0__824___824.png"
    boxes = np.array([[100,200,100,200,0.1]])
    scores = np.array([0.1])
    labels = np.array([1])
    classnames = ["nnn"]
    draw_rboxes(img_file,boxes,scores,labels,classnames)

if __name__ == "__main__":
    test()