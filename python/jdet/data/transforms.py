import random
import jittor as jt 
import cv2
import numpy as np
import math
from PIL import Image
from jdet.utils.registry import build_from_cfg,TRANSFORMS
from jdet.models.boxes.box_ops import rotated_box_to_poly_np,poly_to_rotated_box_np,norm_angle

@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms=None):
        self.transforms = []
        if transforms is None:
            transforms = []
        for transform in transforms:
            if isinstance(transform,dict):
                transform = build_from_cfg(transform,TRANSFORMS)
            elif not callable(transform):
                raise TypeError('transform must be callable or a dict')
            self.transforms.append(transform)

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target

@TRANSFORMS.register_module()
class RandomRotateAug:
    def __init__(self, random_rotate_on=False):
        self.random_rotate_on = random_rotate_on
    
    def _rotate_boxes_90(self,target,size):
        w,h = size
        for key in["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes.ndim<2:
                continue
            if "bboxes" in key or "hboxes" in key:
                new_boxes = np.zeros_like(bboxes)
                new_boxes[:,  ::2] = bboxes[:, 1::2] # x = y
                # new_boxes[:, 1::2] = w - bboxes[:, -2::-2] # y = w - x
                new_boxes[:, 1] = w - bboxes[:, 2] # y = w - x
                new_boxes[:, 3] = w - bboxes[:, 0] # y = w - x
                target[key] = new_boxes
                continue

            if "rboxes" in key:
                bboxes  = rotated_box_to_poly_np(bboxes)

            new_bboxes = np.zeros_like(bboxes)
            new_bboxes[:,0::2] = bboxes[:,1::2]
            new_bboxes[:,1::2] = w-bboxes[:,0::2]

            if "rboxes" in key:
                new_bboxes = poly_to_rotated_box_np(new_bboxes)

            target[key]=new_bboxes

    def __call__( self, image, target=None ):
        # (0, 90, 180, or 270)
        if self.random_rotate_on:
            indx = int(random.random() * 100) // 25
            # anticlockwise
            for _ in range(indx):
                if target is not None:
                    self._rotate_boxes_90(target,image.size)
                image = image.rotate(90,expand=True)
            if target is not None:
                target["rotate_angle"]=90*indx

        return image, target

@TRANSFORMS.register_module()
class Resize:
    def __init__(self, min_size, max_size, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if self.keep_ratio:
            # NOTE Mingtao
            if w <= h:
              size = np.clip( size, int(w / 1.5), int(w * 1.5) )
            else:
              size = np.clip( size, int(h / 1.5), int(h * 1.5) )

            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w),1.

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            assert np.abs(oh/h - ow/w)<1e-2
            
        else:
            oh = self.min_size[0]
            ow = self.max_size
        
        return (oh, ow),oh/h

    def _resize_boxes(self,target,size):
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            width,height = target["img_size"]
            new_w,new_h = size
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)

            target[key]=bboxes
    
    def _resize_mask(self,target,size):
        pass

    def __call__(self, image, target=None):
        size,scale_factor = self.get_size(image.size)
        image = image.resize(size[::-1],Image.BILINEAR)
        if target is not None:
            self._resize_boxes(target,image.size)
            target["img_size"] = image.size
            target["scale_factor"] = scale_factor
            target["pad_shape"] = image.size
            target["keep_ratio"] = self.keep_ratio
        return image, target

@TRANSFORMS.register_module()
class RotatedResize(Resize):

    def _resize_boxes(self, target,size):
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes is None or bboxes.ndim!=2:
                continue
            
            if "rboxes" in key:
                bboxes = rotated_box_to_poly_np(bboxes)

            width,height = target["img_size"]
            new_w,new_h = size
            
            bboxes[:,0::2] = bboxes[:,0::2]*float(new_w/width)
            bboxes[:,1::2] = bboxes[:,1::2]*float(new_h/height)

            # clip to border
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, new_w - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, new_h - 1)
            
            if "rboxes" in key:
                bboxes = poly_to_rotated_box_np(bboxes)
            target[key]=bboxes


@TRANSFORMS.register_module()
class RandomFlip:
    def __init__(self, prob=0.5,direction="horizontal"):
        assert direction in ['horizontal', 'vertical', 'diagonal'],f"{direction} not supported"
        self.direction = direction
        self.prob = prob

    def _flip_boxes(self,target,size):
        w,h = target["img_size"] 
        for key in ["bboxes","polys"]:
            if key not in target:
                continue
            bboxes = target[key]
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

    def _flip_image(self,image):
        if self.direction=="horizontal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.direction == "vertical":
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.direction == "diagonal":
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = self._flip_image(image)
            if target is not None:
                self._flip_boxes(target,image.size)
            target["flip"]=self.direction
        return image, target

@TRANSFORMS.register_module()
class RotatedRandomFlip(RandomFlip):
    def _flip_rboxes(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::5] = w - flipped[..., 0::5] - 1
            flipped[..., 4::5] = norm_angle(np.pi - flipped[..., 4::5])
        elif self.direction == 'vertical':
            assert False
        elif self.direction == 'diagonal':
            assert False
        return flipped

    def _flip_polys(self,bboxes,w,h):
        flipped = bboxes.copy()
        if self.direction == 'horizontal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
        elif self.direction == 'vertical':
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        elif self.direction == 'diagonal':
            flipped[..., 0::2] = w - flipped[..., 0::2] - 1
            flipped[..., 1::2] = h - flipped[..., 1::2] - 1
        return flipped 


    def _flip_boxes(self,target,size):
        w,h = size 
        for key in ["bboxes","hboxes","rboxes","polys","hboxes_ignore","polys_ignore","rboxes_ignore"]:
            if key not in target:
                continue
            bboxes = target[key]
            if "rboxes" in key:
                target[key] = self._flip_rboxes(bboxes,w,h)
                continue 
            if "polys" in key:
                target[key] = self._flip_polys(bboxes,w,h)
                continue
            flipped = bboxes.copy()
            if self.direction == 'horizontal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
            elif self.direction == 'vertical':
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            elif self.direction == 'diagonal':
                flipped[..., 0::4] = w - bboxes[..., 2::4]
                flipped[..., 1::4] = h - bboxes[..., 3::4]
                flipped[..., 2::4] = w - bboxes[..., 0::4]
                flipped[..., 3::4] = h - bboxes[..., 1::4]
            target[key] = flipped

@TRANSFORMS.register_module()
class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def __call__(self,image,target=None):
        if self.size is not None:
            pad_w,pad_h = size
        else:
            pad_h = int(np.ceil(image.size[1] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(image.size[0] / self.size_divisor)) * self.size_divisor
        
        new_image = Image.new(image.mode,(pad_w,pad_h),(self.pad_val,)*len(image.split()))
        new_image.paste(image,(0,0,image.size[0],image.size[1]))
        target["pad_shape"] = (pad_h,pad_w,3)
        
        return new_image,target
    

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self, mean, std, to_bgr=True):
        self.mean = np.float32(mean).reshape(-1,1,1)
        self.std = np.float32(std).reshape(-1,1,1)
        self.to_bgr = to_bgr

    def __call__(self, image, target=None):
        if isinstance(image,Image.Image):
            image = np.array(image).transpose((2,0,1))

        if self.to_bgr:
            image = image[::-1]
        
        image = (image - self.mean) / self.std

        target["mean"] = self.mean 
        target["std"] = self.std
        target["to_bgr"] = self.to_bgr
        return image, target

