from jittor import nn
from jdet.utils.equivalent import FieldType, Rot2dOnR2, regular_feature_type, GeometricTensor
from jdet.models.equivalent_modules import PointwiseAvgPool, PointwiseMaxPool, eReLU, conv1x1, conv3x3, build_norm_layer, R2Conv
from jdet.utils.registry import BACKBONES
import jittor as jt

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='jittor',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None,
                 fixparams=False):
        super(BasicBlock, self).__init__()
        self.in_type = regular_feature_type(gspace, in_channels, fixparams=fixparams)
        self.out_type = regular_feature_type(gspace, out_channels, fixparams=fixparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, gspace, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, gspace, out_channels, postfix=2)

        self.conv1 = conv3x3(
            gspace,
            in_channels,
            self.mid_channels,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            fixparams=fixparams)
        self.__setattr__(self.norm1_name, norm1)
        # self.relu1 = eReLU(self.conv1.out_type, inplace=True)
        self.relu1 = eReLU(self.conv1.out_type, inplace=False)
        self.conv2 = conv3x3(
            gspace,
            self.mid_channels,
            out_channels,
            padding=1,
            bias=False,
            fixparams=fixparams)
        self.__setattr__(self.norm2_name, norm2)
        # self.relu2 = eReLU(self.conv1.out_type, inplace=True)
        self.relu2 = eReLU(self.conv1.out_type, inplace=False)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)
    
    def execute(self, x):

        def _inner_execute(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            raise NotImplementedError
            # out = cp.checkpoint(_inner_execute, x)
        else:
            out = _inner_execute(x)

        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='jittor',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None,
                 fixparams=False):
        super(Bottleneck, self).__init__()
        self.in_type = regular_feature_type(
            gspace, in_channels, fixparams=fixparams)
        self.out_type = regular_feature_type(
            gspace, out_channels, fixparams=fixparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        if self.style == 'jittor':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            raise NotImplementedError

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, gspace, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, gspace, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, gspace, out_channels, postfix=3)

        self.conv1 = conv1x1(
            gspace,
            in_channels,
            self.mid_channels,
            stride=self.conv1_stride,
            bias=False,
            fixparams=fixparams)
        self.__setattr__(self.norm1_name, norm1)
        # self.relu1 = eReLU(self.conv1.out_type, inplace=True)
        self.relu1 = eReLU(self.conv1.out_type, inplace=False)
        self.conv2 = conv3x3(
            gspace,
            self.mid_channels,
            self.mid_channels,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            fixparams=fixparams)

        self.__setattr__(self.norm2_name, norm2)
        # self.relu2 = eReLU(self.conv2.out_type, inplace=True)
        self.relu2 = eReLU(self.conv2.out_type, inplace=False)
        self.conv3 = conv1x1(
            gspace,
            self.mid_channels,
            out_channels,
            bias=False,
            fixparams=fixparams)
        self.__setattr__(self.norm3_name, norm3)
        # self.relu3 = eReLU(self.conv3.out_type, inplace=True)
        self.relu3 = eReLU(self.conv3.out_type, inplace=False)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def execute(self, x):

        def _inner_execute(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu2(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            # out = cp.checkpoint(_inner_execute, x)
            raise NotImplementedError
        else:
            out = _inner_execute(x)

        out = self.relu3(out)

        return out

def get_expansion(block, expansion=None):
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')
    return expansion

class ResLayer(nn.Sequential):
    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 gspace=None,
                 fixparams=False,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                in_type = regular_feature_type(
                    gspace, in_channels, fixparams=fixparams)
                downsample.append(
                    PointwiseAvgPool(
                        in_type,
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True))
            downsample.extend([
                conv1x1(gspace, in_channels, out_channels,
                        stride=conv_stride, bias=False),
                build_norm_layer(norm_cfg, gspace, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gspace=gspace,
                fixparams=fixparams,
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    gspace=gspace,
                    fixparams=fixparams,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

@BACKBONES.register_module()
class ReResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='jittor',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 orientation=8,
                 fixparams=False,
                 pretrained=None):
        super(ReResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        # self.expansion: int
        self.expansion = get_expansion(self.block, expansion)

        self.orientation = orientation
        self.fixparams = fixparams
        self.gspace = Rot2dOnR2(orientation)
        self.in_type = FieldType(
            self.gspace, [self.gspace.trivial_repr] * 3)

        self._make_stem_layer(self.gspace, in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                gspace=self.gspace,
                fixparams=self.fixparams)
            _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.__setattr__(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels
        self.pretrained = pretrained

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, gspace, in_channels, stem_channels):
        if not self.deep_stem:
            in_type = FieldType(
                gspace, in_channels * [gspace.trivial_repr])
            out_type = regular_feature_type(gspace, stem_channels)
            self.conv1 = R2Conv(in_type, out_type, 7,
                                    stride=2,
                                    padding=3,
                                    bias=False,
                                    sigma=None,
                                    frequencies_cutoff=lambda r: 3 * r)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, gspace, stem_channels, postfix=1)
            self.__setattr__(self.norm1_name, norm1)
            # self.relu = eReLU(self.conv1.out_type, inplace=True)
            self.relu = eReLU(self.conv1.out_type, inplace=False)
        self.maxpool = PointwiseMaxPool(
            self.conv1.out_type, kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if not self.deep_stem:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.stop_grad()

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.stop_grad()

    def init_weights(self):
        if self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            print("loading config from {} ...".format(self.pretrained))
            self.load(self.pretrained)

    def execute(self, x):
        if not self.deep_stem:
            x = GeometricTensor(x, self.in_type)
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self):
        super(ReResNet, self).train()
        self._freeze_stages()
        if self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm):
                    m.eval()

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

