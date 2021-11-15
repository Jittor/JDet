from jdet.data.dota import DOTADataset
from jdet.utils.registry import DATASETS
from jdet.config.constant import SSDD_CLASSES

@DATASETS.register_module()
class SSDDDataset(DOTADataset):
    def __init__(self,*arg,**kwargs):
        super().__init__(*arg,**kwargs)
        self.CLASSES = SSDD_CLASSES