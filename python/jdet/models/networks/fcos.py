from jdet.utils.registry import MODELS
from .single_stage import SingleStageDetector

@MODELS.register_module()
class FCOS(SingleStageDetector):
    pass