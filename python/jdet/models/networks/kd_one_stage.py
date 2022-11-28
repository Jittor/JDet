import jittor as jt 
from jittor import nn 
from jdet.utils.general import multi_apply

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS
from jdet.config.config import Config
# from jdet.models.networks import RotatedRetinaNet  
from .rotated_retinanet import RotatedRetinaNet

from pathlib import Path

@MODELS.register_module()
class KnowledgeDistillationSingleStageDetector(RotatedRetinaNet):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Implementation of `Localization Distillation for Object Detection.
    <https://arxiv.org/abs/2204.05957>`_.
    Implementation of `FitNets: Hints for Thin Deep Nets.
    <https://arxiv.org/abs/1412.6550>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """
    def __init__(self,teacher_config,backbone,neck=None,bbox_head=None,teacher_ckpt=None,eval_teacher=True):
        super(KnowledgeDistillationSingleStageDetector,self).__init__(backbone=backbone, neck=neck, bbox_head=bbox_head)
        self.eval_teacher = eval_teacher
        
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            _cfg = Config()
            _cfg.load_from_file(teacher_config)
            teacher_config = _cfg
        self.teacher_model = build_from_cfg(teacher_config['model'], MODELS)
        
        # Load teacher model from checkpoint file
        if teacher_ckpt is not None:
            resume_data = jt.load(teacher_ckpt)
            if ("model" in resume_data):
                self.teacher_model.load_parameters(resume_data["model"])
            elif ("state_dict" in resume_data):
                self.teacher_model.load_parameters(resume_data["state_dict"])
            else:
                self.teacher_model.load_parameters(resume_data)

    def train(self):
        super().train()
        self.backbone.train()
        
    def execute_train(self, images, targets):
        features = self.backbone(images)
        if self.neck:
            features = self.neck(features)
        with jt.no_grad():
            backbonefeatures_teacher = self.teacher_model.backbone(images)
            neckfeatures_teacher = self.teacher_model.neck(backbonefeatures_teacher)
            logits_teacher = multi_apply(self.teacher_model.bbox_head.forward_single, neckfeatures_teacher, self.teacher_model.bbox_head.anchor_strides)
        return self.bbox_head.execute_train(features, neckfeatures_teacher, logits_teacher, targets)

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            outputs: train mode will be losses val mode will be results
        '''
        if self.is_training():
            return self.execute_train(images, targets)
        return super().execute(images, targets)
    
    def train(self):
        super().train()
        self.backbone.train()
        if self.eval_teacher:
            self.teacher_model.eval()
        else:
            self.teacher_model.train()
    
    def dfs(self, parents, k, callback, callback_leave=None):
        ''' An utility function to traverse the module. '''
        n_children = 0
        for name,v in self.__dict__.items():
            if name != "teacher_model" and isinstance(v, nn.Module):
                n_children += 1
        ret = callback(parents, k, self, n_children)
        if ret == False: return
        for k,v in self.__dict__.items():
            if k == "teacher_model":
                continue
            if not isinstance(v, nn.Module):
                continue
            parents.append(self)
            v.dfs(parents, k, callback, callback_leave)
            parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)
