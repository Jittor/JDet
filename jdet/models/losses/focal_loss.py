import jittor as jt 
from jittor import nn

def sigmoid_focal_loss(pred, target, gamma=2.0, alpha=0.25,weight=None,reduction="mean"):
    assert pred.ndim==2 and target.ndim==1 and pred.shape[0]==target.shape[0]
    assert weight is None or (weight.shape[0]==pred.shape[1] and weight.ndim==1)

    num_classes = pred.shape[1]
    class_range = jt.index((1, num_classes),dim=1)+1

    t = target.unsqueeze(1)
    p = pred.sigmoid()
    term1 = (1 - p) ** gamma * jt.log(p)
    term2 = p ** gamma * jt.log(1 - p)
    loss =  -(t == class_range).float() * term1 * alpha - ((t != class_range) & (t >= 0)).float() * term2 * (1 - alpha)

    if weight is not None:
        weight = weight[target]
        loss *= weight.unsqueeze(1)

    if reduction=="mean":
        loss = loss.sum()/pred.shape[0]
    elif reduction == "sum":
        loss  = loss.sum()
    return loss 


def softmax_focal_loss(pred, target, gamma=2.0, alpha=0.25,weight=None,reduction="mean"):
    assert pred.ndim==2 and target.ndim==1 and pred.shape[0]==target.shape[0]
    assert weight is None or (weight.shape[0]==pred.shape[1] and weight.ndim==1)

    p = nn.softmax(pred,dim=1)
    p = p[jt.index((target.shape[0],),dim=0),target]

    term1 = (1 - p) ** gamma * jt.log(p)
    loss =  -(target>=0).float() * term1 * alpha

    if weight is not None:
        weight = weight[target]
        loss *= weight

    if reduction=="mean":
        loss = loss.sum()/pred.shape[0]
    elif reduction == "sum":
        loss  = loss.sum()
    return loss 





def test_focal_loss():
    import torch 
    import numpy as np 
    from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
    from mmcv.ops import softmax_focal_loss as _softmax_focal_loss
    jt.flags.use_cuda=1
    pred = np.random.randn(1024,81).astype(np.float32)
    target = np.random.randint(81,size=(1024,)).astype(np.int32)
    weight = np.random.randn(81).astype(np.float32)

    loss1 = sigmoid_focal_loss(jt.array(pred),jt.array(target),gamma=2.0,alpha=0.25,weight=jt.array(weight),reduction="sum")
    loss2 = _sigmoid_focal_loss(torch.from_numpy(pred).cuda(),torch.from_numpy(target).long().cuda(),2.0,0.25,torch.from_numpy(weight).cuda(),"sum")
    print(loss1,loss2)

    loss1 = softmax_focal_loss(jt.array(pred),jt.array(target),gamma=2.0,alpha=0.25,weight=jt.array(weight),reduction="sum")
    loss2 = _softmax_focal_loss(torch.from_numpy(pred).cuda(),torch.from_numpy(target).long().cuda(),2.0,0.25,torch.from_numpy(weight).cuda(),"sum")
    print(loss1,loss2)

test_focal_loss()