from copy import deepcopy
from pathlib import Path
from jittor import nn 
import jittor as jt
import math

from jdet.models.utils.yolo_modules import *
from jdet.models.boxes.box_ops import bbox_iou_per_box
from jdet.utils.general import make_divisible, check_img_size
from jdet.utils.registry import MODELS
from jdet.data.yolo import non_max_suppression

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, deepcopy(v))

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model)
        self.ema.eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.stop_grad()

    def update(self, model):
        # Update EMA parameters
        with jt.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype == "float32":
                    v *= d
                    v += (1. - d) * msd[k].detach()
            jt.sync_all()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

class ModelEMAWraper(nn.Module):
    def __init__(self, path, **kwargs):
        # Create EMA
        super().__init__()
        self.model = _yolo(path, **kwargs)
        self.ema_hooked = False

    def hook_ema(self):
        self.ema = ModelEMA(self.model)
        self.ema_hooked = True
        print("EMA enabled")
        
    def execute(self, x, targets=None):
        if self.model.is_training():
            if self.ema_hooked:
                self.ema.update(self.model)
            return self.model(x, targets)
        else:
            if self.ema_hooked:
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
                return self.ema.ema(x, targets)
            else:
                return self.model(x, targets)
    
    def state_dict(self):
        if self.ema_hooked:
            return self.ema.ema.state_dict()
        else:
            return self.model.state_dict()
    
    def load_parameters(self, data):
        self.model.load_parameters(data)
        if self.ema_hooked:
            # rehook ema 
            self.ema = ModelEMA(self.model)


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True)
    

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = jt.diag(bn.weight/(jt.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.assign(jt.matmul(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = jt.zeros((conv.weight.shape[0],)) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight*bn.running_mean/jt.sqrt(bn.running_var + bn.eps)
    fusedconv.bias.assign(jt.matmul(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv3 Detect() module m, and correct if necessary
    def sign(x):
        x = jt.ternary(x>0,1,x)
        x = jt.ternary(x<0,-1,x)
        return x 

    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if sign(da) != sign(ds):  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def execute(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = jt.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = jt.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
            
class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [jt.zeros((1,))] * self.nl  # init grid
        a = jt.array(anchors).float().view(self.nl, -1, 2)
        self.anchors =  a.stop_grad()  # shape(nl,na,2)
        self.anchor_grid = a.clone().view(self.nl, 1, -1, 1, 1, 2).stop_grad()  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(*[nn.Conv(x, self.no * self.na, 1) for x in ch])  # output conv

    def execute(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2)

            if not self.is_training():  # inference
                if self.grid[i].ndim<4 or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        
        return x if self.is_training() else (jt.contrib.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = jt.meshgrid([jt.index((ny,),dim=0), jt.index((nx,),dim=0)])
        return jt.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

@MODELS.register_module()
class YOLO(nn.Module):
    #TODO add class weights, autoanchor

    def __init__(
        self, 
        cfg='yolov5s.yaml', 
        ch=3, 
        nc=80,
        imgsz=640,
        anchors=None, 
        boxlg=0.05, # box loss gain
        clslg=0.5,  # class loss gain
        objlg=1.0,  # object loss gain
        cls_pw=1.0, # cls BCELoss positive_weight
        obj_pw=1.0, # obj BCELoss positive_weight
        fl_gamma=0.0, # focal loss gamma
        anchor_t = 4.0, # # anchor-multiple threshold
        single_cls=False,
        conf_thres=0.001,
        is_coco=False):  # model, input channels, number of classes
        
        super().__init__()

        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.nc = self.yaml['nc'] if not single_cls else 1

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        if "names" in self.yaml and len(self.yaml["names"])==self.yaml["nc"]:
            self.names = self.yaml["names"]
        elif single_cls:
            self.names = ['item']
        else:
            self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        assert len(self.names) == self.nc, '%g names found for nc=%g' % (len(self.names), self.nc)


        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = jt.array([s / x.shape[-2] for x in self.forward_once(jt.zeros((1, ch, s, s)))]).int()  # forward
            # m.stride = jt.array([8,16,32]).int()  # forward

            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            print('Strides: %s' % m.stride.tolist())

        gs = int(self.stride.max())  # grid size (max stride)
        nl = self.model[-1].nl
        
        imgsz = check_img_size(imgsz, gs)  # verify imgsz are gs-multiples

        self.box = boxlg * 3. / nl  # scale to layers
        self.cls = clslg * self.nc / 80. * 3. / nl  # scale to classes and layers
        self.obj = objlg * (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.cls_pw = cls_pw
        self.obj_pw = obj_pw
        self.fl_gamma = fl_gamma
        self.anchor_t = anchor_t
        self.conf_thres=conf_thres
        self.is_coco = is_coco
        self.iou_thres=0.65 if is_coco else 0.6

        #initializing hyperparams
        self.gr = 1.0 # iou loss ratio (obj_loss = 1.0 or iou)

        # Init weights, biases
        self.initialize_weights()


    def execute_train(self, x, targets=None):
        # targets.shape = (batch_size, bboxes, labels (1 indexed), masks, classes, img_size, origin_img_size)
        x = self.forward_once(x)
        targets = targets[0]

        losses = self.compute_loss(x, targets)

        return losses

    def execute_test(self, x, labels=None, conf_thres=0.001, iou_thres=0.65):
        inf_out, _ = self.forward_once(x)
        # x = inf_out, train_out
        output = non_max_suppression(inf_out, conf_thres=self.conf_thres, iou_thres=self.iou_thres, labels=[])
        return [output]

    def forward_once(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            #print("input: index:{}, type: {} is {}".format(m.i, m.type, d[0].mean()))
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def execute(self, x, annos=None):
        if annos is None: 
            return self.forward_once(x)
        if self.is_training():
            return self.execute_train(x, annos)
        else: 
            return self.execute_test(x, annos)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else jt.log(cf / cf.sum())  # cls
            mi.bias.assign(b.view(-1))

    
    def initialize_weights(self):
        for m in self.model.modules():
            t = type(m)
            if t is nn.Conv:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm:
                m.eps = 1e-3
                m.momentum = 0.03

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).transpose(1,0)  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.execute = m.fuseforward  # update forward
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def compute_loss(self, p, targets):  # predictions, targets, model
        lcls, lbox, lobj = jt.zeros((1,)), jt.zeros((1,)), jt.zeros((1,))
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=jt.array([self.cls_pw]))  # weight=model.class_weights)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=jt.array([self.obj_pw]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Focal loss
        g = self.fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Losses
        balance = [4.0, 1.0, 0.4, 0.1]  # P3-P6
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = jt.zeros_like(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = jt.contrib.concat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou_per_box(pbox.transpose(1,0), tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).cast(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = jt.full_like(ps[:, 5:], cn)  # targets
                    t[list(range(n)), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in jt.contrib.concat((txy[i], twh[i]), 1)]

            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        lbox *= self.box
        lobj *= self.obj
        lcls *= self.cls
        bs = tobj.shape[0]  # batch size
        # return total_loss * bs, dict(box_loss=lbox * bs, obj_loss=lobj * bs, cls_loss=lcls * bs)
        return dict(box_loss=lbox * bs, obj_loss=lobj * bs, cls_loss=lcls * bs)

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        det = self.model[-1]  # Detect() module
        na, nt = det.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = jt.ones((7,))  # normalized to gridspace gain
        ai = jt.index((na,),dim=0).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        
        targets = jt.contrib.concat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        off = jt.array([[0, 0],
                            # [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ],).float() * g  # offsets

        for i in range(det.nl):
            anchors = det.anchors[i]
            gain[2:6] = jt.array([p[i].shape[3],p[i].shape[2],p[i].shape[3],p[i].shape[2]])  # xyxy gain
            
            # Match targets to anchors
            t = targets * gain

            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = jt.maximum(r, 1. / r).max(2) < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > self.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[jt.array([2, 3])] - gxy  # inverse
                # j, k = jt.logical_and((gxy % 1. < g), (gxy > 1.)).int().transpose(1,0).bool()
                # l, m = jt.logical_and((gxi % 1. < g),(gxi > 1.)).int().transpose(1,0).bool()
                jk = jt.logical_and((gxy % 1. < g), (gxy > 1.))
                lm = jt.logical_and((gxi % 1. < g),(gxi > 1.))
                j, k = jk[:,0],jk[:,1]
                l, m = lm[:,0],lm[:,1]

                j = jt.stack((jt.ones_like(j),))
                t = t.repeat((off.shape[0], 1, 1))[j]
                offsets = (jt.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b = t[:,0].int32()
            c = t[:,1].int32()  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).int32()
            gi, gj = gij[:,0],gij[:,1]  # grid xy indices

            # Append
            a = t[:, 6].int32()  # anchor indices
            indices.append((b, a, gj.clamp(0, gain[3] - 1), gi.clamp(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(jt.contrib.concat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch



def parse_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck ,SPP, DWConv, Focus, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)



def _yolo(cfg, **kwargs):
    model = YOLO(cfg=cfg, **kwargs)
    return model

@MODELS.register_module()
def YOLOv5S(ema=True, **kwargs):
    path = Path(__file__).parent / "../../../../projects/yolo/configs/yolo_configs/yolov5s.yaml"
    model = ModelEMAWraper(path, **kwargs) 
    if ema:
        model.hook_ema()
    return model

@MODELS.register_module()
def YOLOv5M(ema=True, **kwargs):
    path = Path(__file__).parent / "../../../../projects/yolo/configs/yolo_configs/yolov5m.yaml"
    model = ModelEMAWraper(path, **kwargs)
    if ema:
        model.hook_ema()
    return model

@MODELS.register_module()
def YOLOv5L(ema=True, **kwargs):
    path = Path(__file__).parent / "../../../../projects/yolo/configs/yolo_configs/yolov5l.yaml"
    model = ModelEMAWraper(path, **kwargs)
    if ema:
        model.hook_ema()
    return model

@MODELS.register_module()
def YOLOv5X(ema=True, **kwargs):
    path = Path(__file__).parent / "../../../../projects/yolo/configs/yolo_configs/yolov5x.yaml"
    model = ModelEMAWraper(path, **kwargs)
    if ema:
        model.hook_ema()
    return model