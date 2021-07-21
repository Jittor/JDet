import jittor as jt
import torch
box1 = [[[462., 491., 1023., 667.], [335, 361, 357, 415]]]
a = jt.array(box1)
b = jt.full((1, 237136, 4), 500)
c = torch.tensor(box1)
d = torch.full((1, 237136, 4), 500)

ans_jittor = jt.maximum(a[..., :, :2].unsqueeze(-2), b[..., :, :2].unsqueeze(-3))
ans_torch = torch.maximum(c[..., :, :2].unsqueeze(-2), d[..., :, :2].unsqueeze(-3))
print(ans_jittor.sum())
print(ans_torch.sum())

ans1 = jt.maximum(a[..., :, :2].unsqueeze(-2), b[..., 1:10000, :2].unsqueeze(-3))
ans2 = jt.maximum(a[..., :, :2].unsqueeze(-2), b[..., 10000:200000, :2].unsqueeze(-3))
ans3 = jt.maximum(a[..., :, :2].unsqueeze(-2), b[..., 1:200000, :2].unsqueeze(-3))
print(ans1.sum() + ans2.sum())
print(ans3.sum())
print(jt.__version__)