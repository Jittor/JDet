import jittor as jt 
from jittor import nn 

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class RCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self,backbone,neck=None,rpn=None,bbox_head=None):
        super(RCNN,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.rpn = build_from_cfg(rpn,HEADS)
        self.bbox_head = build_from_cfg(bbox_head,HEADS)

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            results: detections
            losses (dict): losses
        '''
        features = self.backbone(images)
        
        if self.neck:
            features = self.neck(features)

        proposals_list, rpn_losses = self.rpn(features,targets)

        # Test code begin

        # import pickle
        # print("load feature")
        # with open('/mnt/disk/czh/masknet/temp/feature.pkl', 'rb') as f:
        #     features_load = pickle.load(f)

        # print("load proposals")
        # proposals_list = []

        # for i in range(len(targets)):

        #     with open(f'/mnt/disk/czh/masknet/temp/proposal_{i}.pkl', 'rb') as f:
        #         proposal = jt.array(pickle.load(f))
        #     with open(f'/mnt/disk/czh/masknet/temp/label_{i}.pkl', 'rb') as f:
        #         label = jt.array(pickle.load(f))
        #     with open(f'/mnt/disk/czh/masknet/temp/obboxes_{i}.pkl', 'rb') as f:
        #         obboxes = jt.array(pickle.load(f))
        #     with open(f'/mnt/disk/czh/masknet/temp/bboxes_{i}.pkl', 'rb') as f:
        #         bboxes = jt.array(pickle.load(f))

        #     proposals_list.append(proposal)
        #     targets[i]['labels'] = label
        #     targets[i]['hboxes'] = bboxes
        #     targets[i]['polys'] = obboxes

        # features = (jt.array(features_load[0]), jt.array(features_load[1]), jt.array(features_load[2]), jt.array(features_load[3]), jt.array(features_load[4]))
        
        # Test code end
       
        output = self.bbox_head(features, proposals_list, targets)

        if self.is_training():
            output.update(rpn_losses)

        return output


