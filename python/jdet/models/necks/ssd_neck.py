import jittor as jt
from jittor import nn
from jdet.utils.registry import NECKS
from jdet.models.utils.weight_init import xavier_init,constant_init
from jittor import init

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def execute(self, x):
        x = self.conv(x)
        x = nn.relu(x)
        return x

@NECKS.register_module()
class SSDNeck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 level_strides,
                 level_paddings,
                 l2_norm_scale=20.,
                 last_kernel_size=3):
        super(SSDNeck, self).__init__()
        assert len(out_channels) > len(in_channels)
        assert len(out_channels) - len(in_channels) == len(level_strides)
        assert len(level_strides) == len(level_paddings)
        assert in_channels == out_channels[:len(in_channels)]

        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)

        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]

        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size \
                if i == len(extra_layer_channels) - 1 else 3

            per_lvl_convs = nn.Sequential(
                ConvModule(out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1),
                ConvModule(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding),
                )
            self.extra_layers.append(per_lvl_convs)
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform',bias=0)

    def execute(self, inputs):
        """Forward function."""
        outs = [feat for feat in inputs]
        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])

        feat = outs[-1]
        for layer in self.extra_layers:
            feat = layer(feat)
            outs.append(feat)
        return tuple(outs)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.
        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.eps = eps
        self.scale = scale
        self.weight = jt.ones((self.n_dims,)) * self.scale

    def execute(self, x):
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdims=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
