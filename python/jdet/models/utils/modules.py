import jittor as jt
from jittor import nn 
import warnings

from .weight_init import kaiming_init,constant_init

from jdet.utils.registry import BRICKS,build_from_cfg


BRICKS.register_module('zero', module=nn.ZeroPad2d)
BRICKS.register_module('reflect', module=nn.ReflectionPad2d)
BRICKS.register_module('replicate', module=nn.ReplicationPad2d)


BRICKS.register_module('Conv1d', module=nn.Conv1d)
BRICKS.register_module('Conv2d', module=nn.Conv2d)
BRICKS.register_module('Conv3d', module=nn.Conv3d)
BRICKS.register_module('Conv', module=nn.Conv2d)


BRICKS.register_module('BN', module=nn.BatchNorm2d)
BRICKS.register_module('BN1d', module=nn.BatchNorm1d)
BRICKS.register_module('BN2d', module=nn.BatchNorm2d)
BRICKS.register_module('BN3d', module=nn.BatchNorm3d)
# BRICKS.register_module('SyncBN', module=SyncBatchNorm)
BRICKS.register_module('GN', module=nn.GroupNorm)
BRICKS.register_module('LN', module=nn.LayerNorm)
BRICKS.register_module('IN', module=nn.InstanceNorm2d)
BRICKS.register_module('IN1d', module=nn.InstanceNorm1d)
BRICKS.register_module('IN2d', module=nn.InstanceNorm2d)
BRICKS.register_module('IN3d', module=nn.InstanceNorm3d)

BRICKS.register_module("ReLU",module=nn.ReLU)
BRICKS.register_module(module=nn.LeakyReLU)
BRICKS.register_module(module=nn.PReLU)
# BRICKS.register_module(module=nn.RReLU)
BRICKS.register_module(module=nn.ReLU6)
BRICKS.register_module(module=nn.ELU)
BRICKS.register_module(module=nn.Sigmoid)
BRICKS.register_module(module=nn.Tanh)
BRICKS.register_module(module=nn.GELU)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_from_cfg(pad_cfg,BRICKS, padding=padding)

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = build_from_cfg(
            dict(type='Conv2d') if conv_cfg is None else conv_cfg,
            BRICKS,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        # self.transposed = self.conv.transposed
        # self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            if norm_cfg.get("type","BN") == "GN":
                self.gn = build_from_cfg(norm_cfg, BRICKS,num_channels=norm_channels)
                self.norm = "gn"
            else:
                self.bn = build_from_cfg(norm_cfg, BRICKS,in_channels=norm_channels)
                self.norm = "bn"
        else:
            self.norm = "None"

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                self.activate = build_from_cfg(act_cfg_,BRICKS)

        # Use msra init by default
        self.init_weights()

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def execute(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = getattr(self,self.norm)(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x