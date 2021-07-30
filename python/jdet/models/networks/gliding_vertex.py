from .rcnn import RCNN
from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class GlidingVertex(RCNN):
    """
    paper: https://arxiv.org/pdf/1911.09358.pdf
    """ 
