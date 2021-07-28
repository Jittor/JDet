import jittor as jt 
from jittor import nn 
from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS
import numpy as np
import jdet
import copy
from my_utils import get_var

@MODELS.register_module()
class RetinaNet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,rpn_net=None):
        super(RetinaNet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        # self.backbone.train()
        # self.backbone.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        # x = get_var('input_img_batch')
        # y = self.backbone(x)

        # y_jt = y[-1].data
        # y_tf = get_var('resnet_feature_dict_C5').data
        # print(np.abs(y_jt - y_tf).mean())

        self.neck = build_from_cfg(neck,NECKS)
        # self.neck.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        # self.neck.train()
        # features = self.neck(self.backbone(x))

        # y_jt = features[2].data
        # y_tf = get_var('feature_pyramid_P5').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[1].data
        # y_tf = get_var('feature_pyramid_P4').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[0].data
        # y_tf = get_var('feature_pyramid_P3').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[3].data
        # y_tf = get_var('feature_pyramid_P6').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[4].data
        # y_tf = get_var('feature_pyramid_P7').data
        # print(np.abs(y_jt - y_tf).mean())

        self.rpn_net = build_from_cfg(rpn_net,HEADS)
        # self.rpn_net.load("weights/RetinaNet_DOTA_2x_20200915_DOTA_702000model.pk_jt.pk")
        # self.rpn_net.train()
        # cnt = 2
        # for x in features:
        #     cnt += 1
        #     bbox_pred, cls_score = self.rpn_net.execute_single(x)
        #     cls_score = cls_score.sigmoid()
        #     y_tf = get_var('rpn_probs_P'+str(cnt)).data
        #     print(cls_score.shape, y_tf.shape)
        #     y_jt = cls_score.data
        #     print(np.abs(y_jt - y_tf).mean())

        #     y_tf = get_var('rpn_boxes_P'+str(cnt)).data
        #     y_jt = bbox_pred.data
        #     print(y_jt.shape, y_tf.shape)
        #     print(np.abs(y_jt - y_tf).mean())
        # sizes = []
        # for x in features:
        #     sizes.append([x.shape[2], x.shape[3]])
        # anchors = self.rpn_net.anchor_generator.grid_anchors(sizes)
        # anchors = jt.concat(anchors, 0)

        # y_tf = get_var('anchors').data
        # y_jt = anchors.data
        # print(y_jt.shape, y_tf.shape)
        # print(np.abs(y_jt - y_tf).mean())

        # results,losses = self.rpn_net(features, targets)
        # exit(0)

    def draw(self, images, results, targets, out_path):
        for i in range(images.shape[0]):
            img = images[i].data
            img = np.transpose(((img + 2.117904) / 5 * 255), [1,2,0]).astype(np.uint8)
            result = copy.deepcopy(results[i])
            target = targets[i]
            result[0][:, [0,2]] = result[0][:, [0,2]] / target["ori_img_size"][0] * target["img_size"][0]
            result[0][:, [1,3]] = result[0][:, [1,3]] / target["ori_img_size"][1] * target["img_size"][1]
            jdet.utils.visualization.visualize_r_result(img.copy(), result, out_path)
            break

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
        
        # y_jt = features[2].data
        # y_tf = get_var('feature_pyramid_P5').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[1].data
        # y_tf = get_var('feature_pyramid_P4').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[0].data
        # y_tf = get_var('feature_pyramid_P3').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[3].data
        # y_tf = get_var('feature_pyramid_P6').data
        # print(np.abs(y_jt - y_tf).mean())

        # y_jt = features[4].data
        # y_tf = get_var('feature_pyramid_P7').data
        # print(np.abs(y_jt - y_tf).mean())
        results,losses = self.rpn_net(features, targets)

        # print(len(losses))
        # y_jt = losses['roi_cls_loss'].data
        # y_tf = get_var('cls_loss').data
        # print(np.abs(y_jt - y_tf).mean(), y_jt, y_tf)
        # y_jt = losses['roi_loc_loss'].data
        # y_tf = get_var('reg_loss').data
        # print(np.abs(y_jt - y_tf).mean(), y_jt, y_tf)
        # exit(0)
        # self.draw(images, results, targets, "temp.jpg")

        # # draw
        # img = images[0].data
        # img = np.transpose(((img + 2.117904) / 5 * 255), [1,2,0]).astype(np.uint8)
        # import cv2
        # img = cv2.resize(img, (600,600))
        # jdet.utils.visualization.visualize_r_result(img.copy(), results[0])
        # print(losses)
        # exit(0)
        if self.is_training():
            return losses 
        else:
            return results
