
import math
import jittor as jt 
from jittor import nn,init
from jittor.misc import _pair

__all__ = ["ORConv2d","RotationInvariantPooling"]

ARF_CUDA_HEADER = r'''
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
#define CeilDIV(a,b) ((a+b-1)/b)
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;i += blockDim.x * gridDim.x)
template <typename Dtype>
__global__ void ARF_forward_cuda_kernel(
  const long nthreads, 
  const Dtype* weight_data,
  const uint8* indices_data,
  const uint16 nInputPlane,
  const uint16 nOutputPlane,
  const uint8 nOrientation,
  const uint8 nRotation,
  const uint16 nEntry,
  Dtype* output_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    uint16 l = n % nEntry;
    uint16 j = (n / nEntry) % nInputPlane;
    uint16 i = n / nEntry / nInputPlane;
    uint8 k;
    Dtype val = *(weight_data + n);
    for (k = 0; k < nRotation; k++) {
      uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
      Dtype *target = output_data + i * (nRotation * nInputPlane * nEntry)
                                  + k * (nInputPlane * nEntry)
                                  + j * (nEntry)
                                  + index;
      *target = val;
    }
  }
}
template <typename Dtype>
__global__ void ARF_backward_cuda_kernel(
  const long nthreads, 
  const Dtype* gradWeight_data,
  const uint8* indices_data,
  const uint16 nInputPlane,
  const uint16 nOutputPlane,
  const uint8 nOrientation,
  const uint8 nRotation,
  const uint16 nEntry,
  Dtype* weight_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
      uint16 l = n % nEntry;
      uint16 j = (n / nEntry) % nInputPlane;
      uint16 i = n / nEntry / nInputPlane;
      uint8 k;
      Dtype *val = weight_data + n;
      *val = 0;
      for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
          Dtype target = *(gradWeight_data + i * (nRotation * nInputPlane * nEntry)
                                           + k * (nInputPlane * nEntry)
                                           + j * (nEntry)
                                           + index);
          *val = *val + target;
      }
  }
}
'''

ARF_CUDA_SRC=r'''
    @alias(weight,in0)
    @alias(indices,in1)
    @alias(output,out0)
    const uint16 nOutputPlane = weight_shape0;
    const uint16 nInputPlane = weight_shape1;
    const uint8 nOrientation = weight_shape2;
    const uint8 kH = weight_shape3;
    const uint8 kW = weight_shape4;
    const uint8 nRotation = indices_shape3;

    const uint16 nEntry = nOrientation * kH * kW;
    const long output_size = nOutputPlane * nInputPlane * nEntry;

    dim3 grid(std::min(CeilDIV(output_size, 512L), 4096L));
    dim3 block(512);

    ARF_forward_cuda_kernel<<<grid, block, 0>>>(
            output_size,
            weight_p,
            indices_p,
            nInputPlane,
            nOutputPlane,
            nOrientation,
            nRotation,
            nEntry,
            output_p);
'''
ARF_CUDA_GRAD_SRC = r'''
    @alias(indices,in0)
    @alias(gradOutput,in1)
    @alias(gradWeight,out0)
    const uint8 nOrientation = indices_shape0;
    const uint8 kH = indices_shape1;
    const uint8 kW = indices_shape2;
    const uint8 nRotation = indices_shape3;
    const uint16 nOutputPlane = gradOutput_shape0 / nRotation;
    const uint16 nInputPlane = gradOutput_shape1 / nOrientation;

    const uint16 nEntry = nOrientation * kH * kW;
    const long count = nOutputPlane * nInputPlane * nEntry;


    dim3 grid(std::min(CeilDIV(count, 512L), 4096L));
    dim3 block(512);

    ARF_backward_cuda_kernel<<<grid, block, 0>>>(
         count,
         gradOutput_p,
         indices_p,
         nInputPlane,
         nOutputPlane,
         nOrientation,
         nRotation,
         nEntry,
         gradWeight_p);
'''
ARF_CPU_HEADER=r'''
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
template <typename T>
void ARF_forward_cpu_kernel(
  const T* weightData,
  const uint8* indicesData,
  const uint16 nOutputPlane,
  const uint16 nInputPlane,
  const uint8 nOrientation,
  const uint8 kH,
  const uint8 kW,
  const uint8 nRotation,
  T* outputData)
{
  const uint16 nEntry = nOrientation * kH * kW;
  uint16 i, j, l;
  uint8 k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        uint16 weightIndex = i * nInputPlane * nEntry
                             + j * nEntry
                             + l;
        T val = *(weightData + weightIndex);
        // T val = *(weightData++);
        for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
          T *target = outputData + i * (nRotation * nInputPlane * nEntry)
                                 + k * (nInputPlane * nEntry)
                                 + j * (nEntry)
                                 + index;
          *target = val;
        }
      }
    }
  }
}
template <typename T>
void ARF_backward_cpu_kernel(
  const uint8* indicesData,
  const T* gradOutputData,
  const uint16 nOutputPlane,
  const uint16 nInputPlane,
  const uint8 nOrientation,
  const uint8 kH,
  const uint8 kW,
  const uint8 nRotation,
  T* gradInputData)
{
  const uint16 nEntry = nOrientation * kH * kW;
  uint16 i, j, l;
  uint8 k;

#pragma omp parallel for private(i, j, l, k)
  for (i = 0; i < nOutputPlane; i++) {
    for (j = 0; j < nInputPlane; j++) {
      for (l = 0; l < nEntry; l++) {
        uint16 gradInputIndex = i * nInputPlane * nEntry
                                + j * nEntry
                                + l;
        T *val = gradInputData + gradInputIndex;
        // T *val = gradInputData++;
        *val = 0;
        for (k = 0; k < nRotation; k++) {
          uint16 index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
          const T *target = gradOutputData + i * (nRotation * nInputPlane * nEntry)
                                           + k * (nInputPlane * nEntry)
                                           + j * (nEntry)
                                           + index;
          *val = *val + *target;
        }
      }
    }
  }
}
'''
ARF_CPU_SRC=r'''
    @alias(weight,in0)
    @alias(indices,in1)
    @alias(output,out0)
    const uint16 nOutputPlane = weight_shape0;
    const uint16 nInputPlane = weight_shape1;
    const uint8 nOrientation = weight_shape2;
    const uint8 kH = weight_shape3;
    const uint8 kW = weight_shape4;
    const uint8 nRotation = indices_shape3;

    ARF_forward_cpu_kernel(
         weight_p,
         indices_p,
         nOutputPlane,
         nInputPlane,
         nOrientation,
         kH,
         kW,
         nRotation,
         output_p);
'''

ARF_CPU_GRAD_SRC=r'''
    @alias(indices,in0)
    @alias(gradOutput,in1)
    @alias(gradInput,out0)
    const uint8 nOrientation = indices_shape0;
    const uint8 kH = indices_shape1;
    const uint8 kW = indices_shape2;
    const uint8 nRotation = indices_shape3;
    const uint16 nOutputPlane = gradOutput_shape0 / nRotation;
    const uint16 nInputPlane = gradOutput_shape1 / nOrientation;

    ARF_backward_cpu_kernel(
         indices_p,
         gradOutput_p,
         nOutputPlane,
         nInputPlane,
         nOrientation,
         kH,
         kW,
         nRotation,
         gradInput_p);
'''


def arf_forward(input,indices):
    assert input.ndim==5,"only supports a batch of ARFs."
    assert input.dtype=="float32" and indices.dtype=="uint8"
    nOutputPlane,nInputPlane,nOrientation,kH,kW = input.size()
    nRotation = indices.shape[3]

    output_shape = (nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW)
    
    output = jt.code(output_shape,input.dtype,[input,indices],cpu_header=ARF_CPU_HEADER,cpu_src=ARF_CPU_SRC,cuda_header=ARF_CUDA_HEADER,cuda_src=ARF_CUDA_SRC)
    return output 

def arf_backward(indices, grad_output):
    assert indices.ndim==4 and indices.dtype=="uint8" and grad_output.dtype=="float32"

    nOrientation,kH,kW,nRotation = indices.size()
    nOutputPlane = grad_output.shape[0] // nRotation
    nInputPlane = grad_output.shape[1] // nOrientation

    output_shape = (nOutputPlane, nInputPlane, nOrientation, kH, kW)    
    output = jt.code(output_shape,grad_output.dtype,[indices,grad_output],cpu_header=ARF_CPU_HEADER,cpu_src=ARF_CPU_GRAD_SRC,cuda_header=ARF_CUDA_HEADER,cuda_src=ARF_CUDA_GRAD_SRC)
    return output 


RIE_CPU_HEADER=r'''
#define FLT_MAX 3.402823466e+38F
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

template <typename T>
void RIE_forward_cpu_kernel(
  const T* feature_data,
  uint8* mainDirection_data,
  T* aligned_data,
  const uint8 nOrientation,
  const uint16 nBatch,
  const uint16 nFeature)
{
  uint16 i;
  uint16 j;
  uint8 l;
  
  #pragma omp parallel for private(i, j, l)
  for (i = 0; i < nBatch; i++) {
    for (j = 0; j < nFeature; j++) {
      uint8 *direction = mainDirection_data + i * nFeature + j;
      T maxVal = -FLT_MAX;
      for (l = 0; l < nOrientation; l++) {
        T val = *(feature_data + i * (nFeature * nOrientation)
                               + j * (nOrientation)
                               + l);
        if (val > maxVal) {
          maxVal = val;
          *direction = l;
        }
      }
      for (l = 0; l < nOrientation; l++) {
        T src = *(feature_data + i * (nFeature * nOrientation)
                               + j * (nOrientation)
                               + l);
        uint8 alignedIndex = (l - (uint8)*direction + nOrientation) % nOrientation;
        T *target = aligned_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + alignedIndex;
        *target = src;
      }
    }
  }
}

template <typename T>
void RIE_backward_cpu_kernel(
  const uint8* mainDirection_data,
  const T* gradOutput_data,
  const uint8 nOrientation,
  const uint16 nBatch,
  const uint16 nFeature,
  T* gradInput_data)
{
  uint16 i;
  uint16 j;
  uint8 l;

  #pragma omp parallel for private(i, j, l)
  for (i = 0; i < nBatch; i++) {
    for (j = 0; j < nFeature; j++) {
      uint8 direction = *(mainDirection_data + i * nFeature + j);
      for (l = 0; l < nOrientation; l++) {
        T src = *(gradOutput_data + i * (nFeature * nOrientation)
                                  + j * (nOrientation)
                                  + l);
        uint8 alignedIndex = (l + direction) % nOrientation;
        T *target = gradInput_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
        *target = src;
      }
    }
  }
}
'''
RIE_CPU_SRC=r'''
    @alias(feature,in0)
    @alias(mainDirection,out0)
    @alias(aligned,out1)
    memset(aligned_p,0,aligned->size);
    const uint16 nBatch = feature_shape0;
    const uint16 nChannel = feature_shape1;
    const uint16 nFeature = nChannel / nOrientation;
    RIE_forward_cpu_kernel(
        feature_p,
        mainDirection_p,
        aligned_p,
        nOrientation,
        nBatch,
        nFeature);

'''
RIE_CPU_GRAD_SRC=r'''
    @alias(mainDirection,in0)
    @alias(gradOutput,in1)
    @alias(gradInput,out0)
    memset(gradInput_p,0,gradInput->size)

    const uint16 nBatch = mainDirection_shape0;
    const uint16 nFeature = mainDirection_shape1;

    RIE_backward_cpu_kernel(
         mainDirection_p,
         gradOutput_p,
         nOrientation,
         nBatch,
         nFeature,
         gradInput_p);
'''

RIE_CUDA_HEADER=r'''
#define FLT_MAX 3.402823466e+38F
typedef unsigned long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
#define CeilDIV(a,b) ((a+b-1)/b)
#define CUDA_KERNEL_LOOP(i, n)  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;i += blockDim.x * gridDim.x)
template <typename Dtype>
__global__ void RIE_forward_cuda_kernel(
  const uint32 nthreads, 
  const Dtype* feature_data,
  const uint16 nBatch,
  const uint16 nFeature,
  const uint8 nOrientation,
  uint8* mainDirection_data,
  Dtype* aligned_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    const uint16 j = n % nFeature;
    const uint16 i = n / nFeature;
    uint8 l;
    
    uint8 *direction = mainDirection_data + i * nFeature + j;
    Dtype maxVal = -FLT_MAX;
    for (l = 0; l < nOrientation; l++) {
      Dtype val = *(feature_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      if (val > maxVal) {
        maxVal = val;
        *direction = l;
      }
    }
    for (l = 0; l < nOrientation; l++) {
      Dtype src = *(feature_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      uint8 alignedIndex = ((l - (uint8)*direction) + nOrientation) % nOrientation;
      Dtype *target = aligned_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
      *target = src;
    } 
  }
}

template <typename Dtype>
__global__ void RIE_backward_cuda_kernel(
  const uint32 nthreads, 
  const Dtype* aligned_data,
  const uint8* mainDirection_data,
  const uint16 nBatch,
  const uint16 nFeature,
  const uint8 nOrientation,
  Dtype* feature_data) 
{
  CUDA_KERNEL_LOOP(n, nthreads) {
    uint8 l;
    const uint16 j = n % nFeature; 
    const uint16 i = n / nFeature;
    const uint8 direction = *(mainDirection_data + i * nFeature + j);
    for (l = 0; l < nOrientation; l++) {
      Dtype src = *(aligned_data + i * (nFeature * nOrientation)
                                 + j * (nOrientation)
                                 + l);
      uint8 alignedIndex = (l + direction) % nOrientation;
      Dtype *target = feature_data + i * (nFeature * nOrientation)
                                   + j * (nOrientation)
                                   + alignedIndex;
      *target = src;
    }
  }
}
'''
RIE_CUDA_SRC=r'''
    @alias(feature,in0)
    @alias(mainDirection,out0)
    @alias(aligned,out1)
    const uint16 nBatch = feature_shape0;
    const uint16 nChannel = feature_shape1;
    const uint16 nFeature = nChannel / nOrientation;

    cudaMemsetAsync(aligned_p,0,aligned->size);
    const long count = nBatch * nFeature;
    dim3 grid(std::min(CeilDIV(count, 512L), 4096L));
    dim3 block(512);

    RIE_forward_cuda_kernel<<<grid, block, 0>>>(
            count,
            feature_p,
            nBatch,
            nFeature,
            nOrientation,
            mainDirection_p,
            aligned_p);
'''
RIE_CUDA_GRAD_SRC=r'''
    @alias(mainDirection,in0)
    @alias(gradOutput,in1)
    @alias(gradInput,out0)
    cudaMemsetAsync(gradInput_p,0,gradInput->size);

    const uint16 nBatch = mainDirection_shape0;
    const uint16 nFeature = mainDirection_shape1;
    const long count = nBatch * nFeature;

    dim3 grid(std::min(CeilDIV(count, 512L), 4096L));
    dim3 block(512);

    RIE_backward_cuda_kernel<<<grid, block, 0>>>(
            count,
            gradOutput_p,
            mainDirection_p,
            nBatch,
            nFeature,
            nOrientation,
            gradInput_p);
'''
def rie_forward(feature, nOrientation):
    assert feature.ndim==4,"only supports a batch of RIEs."
    assert feature.size(2) == 1 and feature.size(3) == 1, "mH x mW should be 1x1."
    nBatch = feature.size(0)
    nChannel = feature.size(1)
    nFeature = nChannel // nOrientation

    prefix_src= f"const uint8 nOrientation = {nOrientation};"

    output = jt.code([(nBatch, nFeature),feature.shape],["uint8",feature.dtype],[feature],
        cpu_header=RIE_CPU_HEADER,
        cpu_src=prefix_src+RIE_CPU_SRC,
        cuda_header=RIE_CUDA_HEADER,
        cuda_src=prefix_src+RIE_CUDA_SRC)
    return output


def rie_backward(mainDirection, grad_output, nOrientation):
    prefix_src= f"const uint8 nOrientation = {nOrientation};"
    output = jt.code(grad_output.shape,grad_output.dtype,[mainDirection, grad_output],
        cpu_header=RIE_CPU_HEADER,
        cpu_src=prefix_src+RIE_CPU_GRAD_SRC,
        cuda_header=RIE_CUDA_HEADER,
        cuda_src=prefix_src+RIE_CUDA_GRAD_SRC)
    return output


class _ActiveRotatingFilter(jt.Function):
    def execute(self, input, indices):
        indices = indices.uint8()
        self.input = input
        output = arf_forward(input, indices)
        self.indices = indices
        return output

    def grad(self, grad_output):
        indices = self.indices
        input = self.input
        grad_input = arf_backward(indices, grad_output)
        return grad_input, None

class _RotationInvariantEncoding(jt.Function):
    def execute(self, input, nOrientation):
        self.nOrientation = nOrientation
        mainDirection, output = rie_forward(input, nOrientation)
        self.saved_tensors = (input, mainDirection)
        return output, mainDirection

    def grad(self, grad_output,g_tmp=None):
        input, mainDirection = self.saved_tensors
        grad_input = rie_backward(mainDirection, grad_output, self.nOrientation)
        return grad_input, None


rotation_invariant_encoding = _RotationInvariantEncoding.apply
active_rotating_filter = _ActiveRotatingFilter.apply


class ActiveRotatingFilter(nn.Module):
    def __init__(self, indices):
        super(ActiveRotatingFilter, self).__init__()
        self.indices = indices

    def execute(self, input):
        return active_rotating_filter(input, self.indices)

class RotationInvariantEncoding(nn.Module):
    def __init__(self, nOrientation, return_direction=False):
        super(RotationInvariantEncoding, self).__init__()
        self.nOrientation = nOrientation
        self.return_direction = return_direction

    def execute(self, input):
        output,d = rotation_invariant_encoding(input, self.nOrientation)
        if self.return_direction:
            return output,d 
        else:
            return output

class RotationInvariantPooling(nn.Module):
    def __init__(self, nInputPlane, nOrientation=8):
        super(RotationInvariantPooling, self).__init__()
        self.nInputPlane = nInputPlane
        self.nOrientation = nOrientation
        #TODO remove this
        hiddent_dim = int(nInputPlane / nOrientation)
        self.conv = nn.Sequential(
            nn.Conv2d(hiddent_dim, nInputPlane, 1, 1),
            nn.BatchNorm2d(nInputPlane),
        )

    def execute(self, x):
        # TODO remove it
        self.conv.eval()
        # x: [N, c, 1, w]
        ## first, max_pooling along orientation.
        N, c, h, w = x.size()
        x = x.view(N, -1, self.nOrientation, h, w)
        x = x.max(dim=2, keepdims=False) # [N, nInputPlane/nOrientation, 1, w]
        # MODIFIED
        # x = self.conv(x) # [N, nInputPlane, 1, w]
        return x


class ORConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, arf_config=None, stride=1,
            padding=0, dilation=1, groups=1, bias=True):
        self.nOrientation, self.nRotation = _pair(arf_config)
        assert (math.log(self.nOrientation) + 1e-5) % math.log(2) < 1e-3, 'invalid nOrientation {}'.format(self.nOrientation)
        assert (math.log(self.nRotation) + 1e-5) % math.log(2) < 1e-3, 'invalid nRotation {}'.format(self.nRotation)

        super(ORConv2d, self).__init__(
        in_channels, out_channels, kernel_size, 
        stride, padding, dilation, groups, bias)
        self.indices = self.get_indices().stop_grad()

        self.weight = jt.zeros((out_channels, in_channels, self.nOrientation, *self.kernel_size))
        if bias:
            self.bias = jt.zeros((out_channels * self.nRotation,))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.nOrientation
        for k in self.kernel_size:
            n *= k
        init.gauss_(self.weight,0,math.sqrt(2.0 / n))

    def get_indices(self, mode='fast'):
        kernel_indices = {
        1: {
            0: (1,),
            45: (1,),
            90: (1,),
            135: (1,),
            180: (1,),
            225: (1,),
            270: (1,),
            315: (1,)
        },
        3: {
            0: (1,2,3,4,5,6,7,8,9),
            45: (2,3,6,1,5,9,4,7,8),
            90: (3,6,9,2,5,8,1,4,7),
            135: (6,9,8,3,5,7,2,1,4),
            180: (9,8,7,6,5,4,3,2,1),
            225: (8,7,4,9,5,1,6,3,2),
            270: (7,4,1,8,5,2,9,6,3),
            315: (4,1,2,7,5,3,8,9,6)
        }
        }
        delta_orientation = 360 / self.nOrientation
        delta_rotation = 360 / self.nRotation
        kH, kW = self.kernel_size
        indices = jt.zeros((self.nOrientation * kH * kW, self.nRotation)).uint8()
        for i in range(0, self.nOrientation):
            for j in range(0, kH * kW):
                for k in range(0, self.nRotation):
                    angle = delta_rotation * k
                    layer = (i + math.floor(angle / delta_orientation)) % self.nOrientation
                    kernel = kernel_indices[kW][angle][j]
                    indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
        return indices.view(self.nOrientation, kH, kW, self.nRotation)

    def rotate_arf(self):
        return active_rotating_filter(self.weight, self.indices)

    def execute(self, input):
        return nn.conv2d(input, self.rotate_arf(), self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def __repr__(self):
        arf_config = '[{}]'.format(self.nOrientation) \
        if self.nOrientation == self.nRotation \
        else '[{}-{}]'.format(self.nOrientation, self.nRotation)
        s = ('{name}({arf_config} {in_channels}, {out_channels}, kernel_size={kernel_size}'
        ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, arf_config=arf_config, **self.__dict__)


def test_arf():
    jt.flags.use_cuda=1
    import math
    out_channels = 4
    in_channels = 2
    nOrientation = 8
    nRotation = 8
    kernel_size = 3

    def get_indices(nOrientation, nRotation, kernel_size, mode='fast'):
        kernel_indices = {
        1: {
            0: (1,),
            45: (1,),
            90: (1,),
            135: (1,),
            180: (1,),
            225: (1,),
            270: (1,),
            315: (1,)
        },
        3: {
            0: (1,2,3,4,5,6,7,8,9),
            45: (2,3,6,1,5,9,4,7,8),
            90: (3,6,9,2,5,8,1,4,7),
            135: (6,9,8,3,5,7,2,1,4),
            180: (9,8,7,6,5,4,3,2,1),
            225: (8,7,4,9,5,1,6,3,2),
            270: (7,4,1,8,5,2,9,6,3),
            315: (4,1,2,7,5,3,8,9,6)
        }
        }
        delta_orientation = 360 / nOrientation
        delta_rotation = 360 / nRotation
        kH, kW = kernel_size
        indices = jt.zeros((nOrientation * kH * kW, nRotation)).uint8()
        for i in range(0, nOrientation):
            for j in range(0, kH * kW):
                for k in range(0, nRotation):
                    angle = delta_rotation * k
                    layer = (i + math.floor(angle / delta_orientation)) % nOrientation
                    kernel = kernel_indices[kW][angle][j]
                    indices[i * kH * kW + j, k] = int(layer * kH * kW + kernel)
        return indices.view(nOrientation, kH, kW, nRotation)

    input = jt.randn((out_channels, in_channels, nOrientation, kernel_size, kernel_size))
    indices = get_indices(nOrientation, nRotation, (kernel_size, kernel_size))
    output = active_rotating_filter(input, indices)
    print(output.size())
    g1 = jt.grad(output.sum(),input)
    print(g1.mean())
    
    input = jt.randn((8,16,32,32))
    orconv = ORConv2d(16, int(16 / 8), kernel_size=3, padding=1, arf_config=(1, 8))
    output = orconv(input)
    print(output.mean())
    g = jt.grad(output.sum(),input)
    print(g.mean())

def test_rie():
    nOrientation = 8
    input = jt.randn(2,8,1,1)
    output,_ = rotation_invariant_encoding(input, nOrientation)

    g1 = jt.grad(output.sum(),input)
    print(g1.mean())

def test_rip():
    inst = RotationInvariantPooling(512, 8)
    input = jt.randn(8, 512, 1, 25)
    output = inst(input)
    print(output.size())

if __name__ == "__main__":
    test_arf()
    test_rie()
    test_rip()
            