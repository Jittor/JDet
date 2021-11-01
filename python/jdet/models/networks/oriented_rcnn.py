from .rcnn import RCNN
from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class OrientedRCNN(RCNN):
    """
    paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf
    """ 
