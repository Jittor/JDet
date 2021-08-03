import pickle
import numpy as np
import jittor as jt
from jdet.models.losses.focal_loss import sigmoid_focal_loss
jt.flags.use_cuda = True

def sigmoid_cross_entropy_with_logits(logits, labels):
    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.

    relu_logits = jt.ternary(logits >= 0., logits, jt.broadcast_var(0.0, logits))
    neg_abs_logits = -jt.abs(logits)
    return relu_logits - logits * labels + jt.log((neg_abs_logits).exp() + 1)


def binary_cross_entropy_with_logits(output, target):
    max_val = jt.clamp(-output,min_v=0)
    loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()

    return loss


def sigmoid_focal_loss(inputs,targets,weight=None, alpha = -1,gamma = 2,reduction = "none",avg_factor=None):
    
    targets = targets.broadcast(inputs,[1])
    targets = (targets.index(1)+1)==targets
    p = inputs.sigmoid()
    assert(weight is None)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    breakpoint()

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        if avg_factor is None:
            avg_factor = loss.numel()
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss, ce_loss


with open('focal_input.pk', 'rb') as f:
    inputs, cates = pickle.load(f)


def run_jt():
    a = jt.array(inputs)
    b = jt.array(cates)
    opt = jt.nn.SGD([a], lr=5e-3, momentum=0.9)
    for t in range(100):
        l, ce_loss = sigmoid_focal_loss(a,b,reduction="sum",alpha=0.25)
        print(t,l.item())
        yield ce_loss.data
        opt.step(l)

def run_torch():
    targets = jt.array(cates)
    targets = targets.broadcast(inputs,[1])
    targets = (targets.index(1)+1)==targets
    targets = targets.data.astype(np.float32)

    a = torch.from_numpy(inputs)
    a.requires_grad = True
    b = torch.from_numpy(targets)
    opt = torch.optim.SGD([a], lr=5e-3, momentum=0.9)
    for t in range(100):
        opt.zero_grad()
        l, ce_loss = torch_sigmoid_focal_loss(a,b,reduction="sum",alpha=0.25)
        print(t,l.item())
        l.backward()
        yield ce_loss.detach().numpy()
        opt.step()


# run_jt()
# run_torch()

JT = run_jt()
TH = run_torch()
ce_jt = next(JT)
ce_th = next(TH)
print(np.abs(ce_jt - ce_th).sum())
# breakpoint()
