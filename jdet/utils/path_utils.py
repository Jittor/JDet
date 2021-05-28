import os 
import glob 

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
    for img_d in img_dir.split(",")
        if len(img_d)==0:
            continue
        if not os.path.exists(img_d):
            raise f"{img_d} not exists"
        img_d = os.path.abspath(img_d)
        img_files.extend([f for f in list_files(img_d) if is_img(f)])

    return img_files
