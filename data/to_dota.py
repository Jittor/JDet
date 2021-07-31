import pickle
import os
import cv2
from xml.dom.minidom import parse

def solve_xml(src, tar):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    box_list=[]
    for obj in objects:
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        box_list.append({"name":name, "bbox":bbox})
    
    file=open(tar,'w')
    print("imagesource:GoogleEarth",file=file)
    print("gsd:0.0",file=file)
    for box in box_list:
        ss=""
        for f in box["bbox"]:
            ss+=str(f)+" "
        ss+=box["name"]+" 0"
        print(ss,file=file)
    file.close()

os.makedirs("fair_DOTA")

for root, dirs, files in os.walk("fair/images"):
    for f in files:
        src=os.path.join(root, f)
        print("src",src)
        tar="P"+f[:-4].zfill(4)+".png"
        tar=os.path.join("fair_DOTA","images", tar)
        print("tar",tar)

        file = cv2.imread(src, 1)
        cv2.imwrite(tar, file)

for root, dirs, files in os.walk("fair/labelXmls"):
    for f in files:
        src=os.path.join(root, f)
        print("src",src)
        tar="P"+f[:-4].zfill(4)+".txt"
        tar=os.path.join("fair_DOTA","labelTxt", tar)
        print("tar",tar)
        solve_xml(src, tar)
