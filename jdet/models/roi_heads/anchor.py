from jdet.utils.registry import ROI_HEADS

@ROI_HEADS.register_module()
class AnchorGenerator:
    def __init__(self):
        pass