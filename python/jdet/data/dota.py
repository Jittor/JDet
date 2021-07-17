from jdet.data.coco import COCODataset
from jdet.models.boxes.box_ops import rotated_box_to_poly_single
from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES

from .custom import CustomDataset

@DATASETS.register_module()
class DOTADataset(CustomDataset):
    CLASSES = DOTA1_CLASSES
    
@DATASETS.register_module()
class DOTADataset_vCOCO(COCODataset):
    CLASSES = DOTA1_CLASSES