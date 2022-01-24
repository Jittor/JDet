import random

import jittor as jt 
import cv2
import numpy as np
import math
from PIL import Image
from jdet.utils.registry import build_from_cfg,TRANSFORMS
from jdet.models.boxes.box_ops import rotated_box_to_poly_np,poly_to_rotated_box_np,norm_angle
from jdet.models.boxes.iou_calculator import bbox_overlaps_np
from jdet.ops.bbox_move import warp
from jdet.ops.bbox_transforms_numpy import bbox2type
from jdet.ops.bbox_geometry import bbox_overlaps
from numpy import random as nprandom

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
    def __init__(self, 
                 random_rotate_on=False, 
                 rotate_90=True,
                 keep_shape=True,
                 angles=(0, 90),
                 rotate_mode='range',
                 vert_rate=0.5,
                 vert_cls=None,
                 label_standard="rboxes",
                 keep_iof_thr=0.7):

        self.random_rotate_on = random_rotate_on
        self.rotate_90 = rotate_90
        self.keep_shape = keep_shape
        self.angles = angles
        self.rotate_mode = rotate_mode
        self.vert_rate = vert_rate
        self.vert_cls = vert_cls
        self.keep_iof_thr = keep_iof_thr
        self.label_standard = label_standard

    def get_matrix_and_size(self, target):
        angle = target["rotate_angle"]
        height, width = target["img_size"][:2]
        if self.keep_shape:
            center = ((width - 1) * 0.5, (height - 1) * 0.5)
            matrix = cv2.getRotationMatrix2D(center, angle, 1)
        else:
            matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
            img_bbox = np.array([[0, 0, width, 0, width, height, 0, width]])
            img_bbox = bbox2type(warp(img_bbox, matrix), 'hbb')

            width = int(img_bbox[0, 2] - img_bbox[0, 0] + 1)
            height = int(img_bbox[0, 3] - img_bbox[0, 1] + 1)
            matrix[0, 2] = -img_bbox[0, 0]
            matrix[1, 2] = -img_bbox[0, 1]
        return matrix, width, height

    def _rotate_boxes_90(self, target,size):

        w, h = size
        for key in["bboxes", "hboxes", "rboxes", "polys", "hboxes_ignore", "polys_ignore", "rboxes_ignore"]:
            
            if key not in target:
                continue
            bboxes = target[key]
            if bboxes.ndim < 2:
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

            target[key] = new_bboxes

    def _rotate_boxes_rand(self, target, matrix, w, h, img_bound):
        
        for key in ["bboxes", "hboxes", "rboxes", "polys",]:
            if key not in target:
                continue

            bboxes = target[key]
            if bboxes.ndim < 2:
                continue
            warped_bboxes = warp(target[key], matrix, keep_type=True)
            if self.keep_shape:
                iofs = bbox_overlaps(warped_bboxes, img_bound, mode='iof')
                if_inwindow = iofs[:, 0] > self.keep_iof_thr
                new_bboxes = warped_bboxes[if_inwindow]
            
            if key == self.label_standard:
                label_if_inwindow = if_inwindow.copy()

            target[key] = new_bboxes

        if "labels" in target.keys():
            target['labels'] = target['labels'][label_if_inwindow]

    def __call__( self, image, target=None ):

        if self.random_rotate_on:

            vert = False
            if self.vert_cls is not None:
                if "cls" not in target:
                    raise ValueError('need class order when vert_cls is not None')
                vert_lbls = [target["cls"].index(c) for c in self.vert_cls]
                if "labels" in target:
                    labels = target["labels"]
                    for i in vert_lbls:
                        if (labels == i).any():
                            vert = True
        
            vert = True if np.random.rand() < self.vert_rate else vert
            
            if self.rotate_90 or vert == True:
                # (0, 90, 180, or 270)              
                indx = int(random.random() * 100) // 25
                # anticlockwise
                for _ in range(indx):
                    if target is not None:
                        self._rotate_boxes_90(target , image.size)
                    image = image.rotate(90, expand=True)
                if target is not None:
                    target["rotate_angle"] = 90 * indx

            else:

                target["rboxes"][:, 4] *= -1
                if self.rotate_mode == 'value':
                    angles = list(self.angles)
                    angles = angles + [0] if 0 not in angles else angles
                    np.random.shuffle(angles)
                    angle = angles[0]
                else:
                    angle_min, angle_max = min(self.angles), max(self.angles)
                    angle = (angle_max - angle_min) * np.random.rand() + angle_min

                if target is not None:
                    target["rotate_angle"] = angle
                
                if angle != 0:
                    matrix, w, h = self.get_matrix_and_size(target)

                    img_bound = np.array([[0, 0, w, 0, w, h, 0, h]])
                    self._rotate_boxes_rand(target, matrix, w, h, img_bound)
                    image = image.rotate(angle)
                target["rboxes"][:, 4] *= -1

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
class MinIoURandomCrop:
    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, image, target=None):
        boxes = target['bboxes']
        w, h = image.size
        while True:
            mode = nprandom.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return image, target

            min_iou = mode
            for i in range(50):
                new_w = nprandom.uniform(self.min_crop_size * w, w)
                new_h = nprandom.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = nprandom.uniform(w - new_w)
                top = nprandom.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps_np(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    boxes = boxes[mask]
                    if self.bbox_clip_border:
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['bboxes'] = boxes
                    # labels
                    target['labels'] = target['labels'][mask]

                # adjust the img no matter whether the gt is empty before crop
                image_crop = image.crop(patch)
                target['img_size'] = image_crop.size

                return image_crop, target


@TRANSFORMS.register_module()
class Expand:
    def __init__(self,
                 mean=(0, 0, 0),
                 ratio_range=(1, 4),
                 prob=0.5):
        self.ratio_range = ratio_range
        self.mean = tuple([int(i) for i in mean])
        self.min_ratio, self.max_ratio = ratio_range
        self.prob = prob

    def __call__(self, image, target=None):
        if nprandom.uniform(0, 1) > self.prob:
            return image, target
        w, h = image.size
        ratio = nprandom.uniform(self.min_ratio, self.max_ratio)
        left = int(nprandom.uniform(0, w * ratio - w))
        top = int(nprandom.uniform(0, h * ratio - h))
        
        new_image = Image.new(image.mode,(int(w * ratio), int(h * ratio)), self.mean)
        new_image.paste(image,(left, top, left+image.size[0], top+image.size[1]))

        target["bboxes"] = target["bboxes"] + np.tile(
            (left, top), 2).astype(target["bboxes"].dtype)
        target['img_size'] = new_image.size
        return new_image, target

@TRANSFORMS.register_module()
class PhotoMetricDistortion:
    def __init__(self, brightness_delta=32./255, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.t = jt.transform.ColorJitter(brightness=brightness_delta, contrast=contrast_range, saturation=saturation_range, hue=hue_delta)

    def __call__(self, image, target=None):
        image = self.t(image)
        
        return image, target

@TRANSFORMS.register_module()
class Resize_keep_ratio:
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

        oh = self.min_size[0]
        ow = self.max_size
        return (oh, ow), [ow/w, oh/h, ow/w, oh/h]
        

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
            flipped[..., 1::5] = h - flipped[..., 1::5] - 1
            flipped[..., 4::5] = norm_angle( - flipped[..., 4::5])
        elif self.direction == 'diagonal':
            assert False
        else:
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
            pad_w,pad_h = self.size
        else:
            pad_h = int(np.ceil(image.size[1] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(image.size[0] / self.size_divisor)) * self.size_divisor
        
        new_image = Image.new(image.mode,(pad_w,pad_h),(self.pad_val,)*len(image.split()))
        new_image.paste(image,(0,0,image.size[0],image.size[1]))
        target["pad_shape"] = new_image.size
        
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

@TRANSFORMS.register_module()
class FliterEmpty:
    
    def __init__(self, fliter_list):
        self.fliter_list = fliter_list

    def __call__(self, image, target=None):
        
        for k in self.fliter_list:
            if k == "rboxes" or k == "hboxes" or k == "polys" or k == "bboxes":
                if target[k].size == 0:
                    return image, None

        return image, target
