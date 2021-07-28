import jittor as jt 
def nms(boxes,scores,thresh):
    assert boxes.shape[-1]==4 and len(scores)==len(boxes)
    if scores.ndim==1:
        scores = scores.unsqueeze(-1)
    dets = jt.concat([boxes,scores],dim=1)
    return jt.nms(dets,thresh)
