import jittor as jt 
from jittor import nn 

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS
from my_utils import get_var, _show_keys
import numpy as np
import jdet
import cv2

@MODELS.register_module()
class RetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,rpn_net=None,first_conv=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.backbone.eval()
        self.backbone.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        x = get_var('input_img_batch')
        input_img = x
        y = self.backbone(x)
        y_jt = y[-1].data
        y_tf = get_var('resnet_feature_dict_C5').data
        print(np.abs(y_jt - y_tf).mean())

        self.neck = build_from_cfg(neck,NECKS)
        # ks = self.neck.named_parameters()
        # for k in ks:
        #     print(k[0])
        self.neck.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        features = self.neck(self.backbone(x))

        y_jt = features[2].data
        y_tf = get_var('feature_pyramid_P5').data
        print(np.abs(y_jt - y_tf).mean())

        y_jt = features[1].data
        y_tf = get_var('feature_pyramid_P4').data
        print(np.abs(y_jt - y_tf).mean())

        y_jt = features[0].data
        y_tf = get_var('feature_pyramid_P3').data
        print(np.abs(y_jt - y_tf).mean())

        y_jt = features[3].data
        y_tf = get_var('feature_pyramid_P6').data
        print(np.abs(y_jt - y_tf).mean())

        y_jt = features[4].data
        y_tf = get_var('feature_pyramid_P7').data
        print(np.abs(y_jt - y_tf).mean())

        self.rpn_net = build_from_cfg(rpn_net,HEADS)
        self.rpn_net.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        
        cnt = 2
        for x,feat_stride in zip(features,self.rpn_net.feat_strides):
            cnt += 1
            bbox_pred, cls_score = self.rpn_net.execute_single(x,feat_stride)
            cls_score = cls_score.sigmoid()
            y_tf = get_var('rpn_probs_P'+str(cnt)).data
            print(cls_score.shape, y_tf.shape)
            y_jt = cls_score.data
            print(np.abs(y_jt - y_tf).mean())

            y_tf = get_var('rpn_boxes_P'+str(cnt)).data
            y_jt = bbox_pred.data
            print(y_jt.shape, y_tf.shape)
            print(np.abs(y_jt - y_tf).mean())

        
        sizes = []
        for x in features:
            sizes.append([x.shape[2], x.shape[3]])
        anchors = self.rpn_net.anchor_generator.grid_anchors(sizes)
        anchors = jt.concat(anchors, 0)

        y_tf = get_var('anchors').data
        y_jt = anchors.data
        print(y_jt.shape, y_tf.shape)
        print(np.abs(y_jt - y_tf).mean())
        self.rpn_net.eval()
        results,losses = self.rpn_net(features, [{"bboxes":[0,1,2,3,4], "labels":1}])
        print(losses)
        print(results)
        print(len(results))
        img = input_img[0].data
        img = np.transpose(((img + 2.117904) / 5 * 255), [1,2,0]).astype(np.uint8)
        jdet.utils.visualization.visualize_r_result(img.copy(), results[0])
        boxes = get_var('boxes').data
        x, y, w, h, a = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4], boxes[:, 4:5]
        boxes = np.concatenate([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h, a], 1) 
        result = {  'boxes': boxes,
                    'scores': get_var('scores').data,
                    'labels': get_var('labels').data}
        jdet.utils.visualization.visualize_r_result(img.copy(), result,'test1.jpg')
        exit(0)

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
        
        results,losses = self.rpn_net(features, targets)
        
        if self.is_training():
            return losses 
        else:
            return results
