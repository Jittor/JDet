from jdet.utils.equivalent.equivalent_utils import regular_feature_type
from .batchnormalization import InnerBatchNorm
from .upsampling import R2Upsampling
from .relu import eReLU
from .pooling import PointwiseMaxPool
from .e2conv import R2Conv

def build_norm_layer(cfg, gspace, num_features, postfix=''):
    in_type = regular_feature_type(gspace, num_features)
    return 'bn' + str(postfix), InnerBatchNorm(in_type)

def ennInterpolate(gspace, inplanes, scale_factor, mode='nearest', align_corners=False):
    in_type = regular_feature_type(gspace, inplanes)
    return R2Upsampling(in_type, scale_factor, mode=mode, align_corners=align_corners)

def ennReLU(gspace, inplanes, inplace=True):
    in_type = regular_feature_type(gspace, inplanes)
    return eReLU(in_type, inplace=inplace)

def ennMaxPool(gspace, inplanes, kernel_size, stride=1, padding=0):
    in_type = regular_feature_type(gspace, inplanes)
    return PointwiseMaxPool(in_type, kernel_size=kernel_size, stride=stride, padding=padding)

def convnxn(gspace, inplanes, out_planes, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1, fixparams=False):
    in_type = regular_feature_type(gspace, inplanes, fixparams=fixparams)
    out_type = regular_feature_type(gspace, out_planes, fixparams=fixparams)
    return R2Conv(in_type, out_type, kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      dilation=dilation,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)

def conv3x3(gspace, inplanes, out_planes, stride=1, padding=1, dilation=1, bias=False, fixparams=False):
    """3x3 convolution with padding"""
    in_type = regular_feature_type(gspace, inplanes, fixparams=fixparams)
    out_type = regular_feature_type(gspace, out_planes, fixparams=fixparams)
    return R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)


def conv1x1(gspace, inplanes, out_planes, stride=1, padding=0, dilation=1, bias=False, fixparams=False):
    """1x1 convolution"""
    in_type = regular_feature_type(gspace, inplanes, fixparams=fixparams)
    out_type = regular_feature_type(gspace, out_planes, fixparams=fixparams)
    return R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3 * r)
