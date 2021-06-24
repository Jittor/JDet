from jdet.utils.registry import DATASETS
from jdet.config.constant import DOTA1_CLASSES

from .custom import CustomDataset

@DATASETS.register_module()
class DOTADataset(CustomDataset):
    CLASSES = DOTA1_CLASSES
    