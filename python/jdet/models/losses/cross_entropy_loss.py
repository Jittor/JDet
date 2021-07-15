import jittor as jt

def cross_entropy_loss(output, target, weight=None, ignore_index=None):
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = jt.ones(target.shape[0], dtype='float32')
    if weight is not None:
        target_weight = weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target == ignore_index,
            jt.array(0).broadcast(target_weight),
            target_weight
        )

    target = target.broadcast(output, [1])
    target = target.index(1) == target

    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight

    return loss / target_weight