from PIL import Image,ImageDraw
# import jittor as jt 
import numpy as np
import random
import cv2
from jdet.data import transforms
from jdet.utils.registry import build_from_cfg,TRANSFORMS

# a = torch.zeros((256,512,1,1,))
# torch.nn.init.xavier_uniform_(a,1)
# print(a.mean(),a.std())

# b = jt.zeros((256,512,1,1,))
# jt.nn.init.xavier_uniform_(b,1)
# print(b.mean(),b.std())

def array2image(img):
    return  Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def image2array(img):
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    return img

def draw_poly(img,point,color,thickness):
    cv2.line(img, tuple(point[0]), tuple(point[1]), color, thickness)
    cv2.line(img, tuple(point[1]), tuple(point[2]), color, thickness)
    cv2.line(img, tuple(point[2]), tuple(point[3]), color, thickness)
    cv2.line(img, tuple(point[3]), tuple(point[0]), color, thickness)

def draw(image,target,filename="t1.jpg"):
    image = image2array(image)
    boxes = target["hboxes"][0].astype(int)
    x1,y1,x2,y2 = boxes
    cv2.rectangle(image,(x1, y1),(x2, y2),color=(255,255,0),thickness=1) 

    boxes = target["polys"][0]
    points = boxes.reshape(4,2)
    draw_poly(image,points,color=(255,255,0),thickness=1)
    cv2.imwrite(filename,image)

t = build_from_cfg(
    dict(
        type="Compose",
        transforms=[ 
            dict(
                type='RotatedRandomFlip', 
                prob=1,
            ),
            # dict(
            #     type="RandomRotateAug",
            #     random_rotate_on=True
            # )

        ]
    ),
    TRANSFORMS
)


image = Image.open("test.jpg")
bboxes = np.array([[100.,200.,120,300]])
polys = np.array([[200,300,400,500,230,450,400,700]])
target = dict(
         hboxes=bboxes,
         polys=polys,
         img_size=image.size
         )
draw(image,target,"t1.jpg")
image,target = t(image,target)
draw(image,target,"t2.jpg")
