import jittor as jt
import jittor.nn as nn

from jittor.nn import _pair
from jdet.ops.san_aggregations import aggregation
from jdet.ops.san_subtractions import subtraction, subtraction2
from jdet.utils.registry import MODELS, LOSSES, build_from_cfg
from jdet.models.losses import SANMixUpLoss


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def position(H, W):
    loc_w = jt.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
    loc_h = jt.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = jt.concat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

class Subtraction(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def execute(self, input):
        return subtraction(input, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)

class Subtraction2(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Subtraction2, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def execute(self, input1, input2):
        return subtraction2(input1, input2, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)

class Aggregation(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation, pad_mode):
        super(Aggregation, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def execute(self, input, weight):
        return aggregation(input, weight, self.kernel_size, self.stride, self.padding, self.dilation, self.pad_mode)

class Unfold(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super(Unfold, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
    
    def execute(self, x):
        return nn.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

class SAM(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.sa_type, self.kernel_size, self.stride = sa_type, kernel_size, stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        if sa_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2), nn.ReLU(),
                                        nn.Conv2d(rel_planes + 2, rel_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes), nn.ReLU(),
                                        nn.Conv2d(rel_planes, out_planes // share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(),
                                        nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes // share_planes, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(out_planes // share_planes), nn.ReLU(),
                                        nn.Conv2d(out_planes // share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size - 1) + 1) // 2, dilation, pad_mode=1)

    def execute(self, x):
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.sa_type == 0:  # pairwise
            p = self.conv_p(position(x.shape[2], x.shape[3]))
            w = self.softmax(self.conv_w(jt.concat([self.subtraction2(x1, x2), self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        else:  # patchwise
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.reshape((x.shape[0], -1, 1, x.shape[2]*x.shape[3]))
            x2 = self.unfold_j(self.pad(x2)).reshape((x.shape[0], -1, 1, x1.shape[-1]))
            w = self.conv_w(jt.concat([x1, x2], 1)).reshape((x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1]))
        x = self.aggregation(x3, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sam = SAM(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU()
        self.stride = stride

    def execute(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

@MODELS.register_module()
class SAN(nn.Module):
    def __init__(self, sa_type, layers, kernels, num_classes, block=Bottleneck, loss=None, loss_prepare=False):
        super(SAN, self).__init__()
        self.loss = build_from_cfg(loss, LOSSES)
        self.loss_prepare = loss_prepare
        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(sa_type, block, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c // 4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(sa_type, block, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(sa_type, block, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(sa_type, block, c, layers[3], kernels[3])

        c *= 2
        self.conv4, self.bn4 = conv1x1(c // 2, c), nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(sa_type, block, c, layers[4], kernels[4])

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=7, stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def execute(self, x, targets=None):
        targets = jt.array([t['img_label'] for t in targets])
        if self.is_training() and self.loss_prepare:
            x = self.loss.prepare(x, targets)

        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        if self.is_training():
            return dict(loss=(self.loss(x, targets)))
        return x


def san(sa_type, layers, kernels, num_classes):
    model = SAN(sa_type, layers, kernels, num_classes, block=Bottleneck, loss=dict(type='SAMSmoothLoss'), loss_prepare=True)
    return model


if __name__ == '__main__':
    jt.flags.use_cuda=1
    net = san(sa_type=0, layers=(3, 4, 6, 8, 3), kernels=[3, 7, 7, 7, 7], num_classes=1000)
    targets = [dict(img_label=1),dict(img_label=2),dict(img_label=3),dict(img_label=4)]
    y = net(jt.randn(4, 3, 224, 224), targets)
    print(y)
    net.eval()
    y = net(jt.randn(4, 3, 224, 224), targets)
    print(y.size())
