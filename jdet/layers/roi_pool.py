from jittor import nn,init
import jittor as jt
from jittor.misc import _pair

__all__ = ["ROIPool"]


CUDA_HEADER = r'''
#include <cmath>
#include <cstdio>
#include <climits>
using namespace std;
'''

CUDA_SRC = r'''
__global__ static void RoIPoolForwardKernel(@ARGS_DEF){
    @PRECALC
@alias(input,in0);
@alias(rois,in1);
@alias(output,out0);
@alias(argmax,out1);
const int pooled_height = output_shape2;
const int pooled_width = output_shape3;
const int  num_rois = rois_shape0;
const int  channels = input_shape1;
const int height = input_shape2;
const int width = input_shape3;
const int  nthreads = num_rois * pooled_height * pooled_width * channels;
const float spatial_scale = @in2(0);
auto bottom_data =  input_p;
auto bottom_rois = rois_p;
auto top_data = output_p;
auto argmax_data = argmax_p;
 for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    auto offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);
    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    float bin_size_h = 1.0*roi_height / pooled_height;
    float bin_size_w = 1.0*roi_width / pooled_width;
    
    int hstart = static_cast<int>(floor(ph* bin_size_h));
    int wstart = static_cast<int>(floor(pw* bin_size_w));
    int hend = static_cast<int>(ceil((ph + 1)* bin_size_h));
    int wend = static_cast<int>(ceil((pw + 1)* bin_size_w));
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);
    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -99999999;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    auto offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
 
}
cudaMemsetAsync(argmax_p,0,argmax->size);
cudaMemsetAsync(output_p,0,output->size);
const int total_count = in1_shape0 * out0_shape2 * out0_shape3 * in0_shape1;
const int thread_per_block = 1024;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
RoIPoolForwardKernel<<<block_count, thread_per_block>>>(@ARGS);
'''

CUDA_GRAD_SRC=[r'''
__global__ void RoIPoolBackwardKernel(@ARGS_DEF){
    @PRECALC
    @alias(input,in0);
    @alias(rois,in1);
    @alias(argmax,pout0)
    @alias(grad_input,out0);
    @alias(grad,dout);
    const float spatial_scale = @in2(0);
    const int pooled_height = pout0_shape2;
    const int pooled_width = pout0_shape3;
    const int num_rois = rois_shape0;
    const int channels = input_shape1;
    const int height = input_shape2;
    const int width = input_shape3;
    const int nthreads = num_rois * pooled_height * pooled_width * channels;
    auto top_diff = grad_p;
    auto argmax_data = argmax_p;
    auto bottom_diff = grad_input_p;
    auto bottom_rois = rois_p;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    auto offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    auto offset_top_diff = top_diff + top_offset;
    auto offset_bottom_diff = bottom_diff + bottom_offset;
    auto offset_argmax_data = argmax_data + top_offset;
    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      atomicAdd(
          offset_bottom_diff + argmax,
          offset_top_diff[ph * pooled_width + pw]);
    }
  }
}
cudaMemsetAsync(out0_p,0,out0->size);
const int total_count = in1_shape0*in0_shape1*pout0_shape2*pout0_shape3;
const int thread_per_block = 1024;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
RoIPoolBackwardKernel<<<block_count, thread_per_block>>>(@ARGS);
''',r'''
''',
r'''
''']

def roi_pool(input,rois,output_size,spatial_scale):
    output_size = _pair(output_size)
    spatial_scale = jt.array([spatial_scale])
    output_shapes = [(rois.shape[0], input.shape[1], output_size[0], output_size[1])]*2
    inputs = [input,rois,spatial_scale]
    output_types = [input.dtype,'int32']
    output,arg_output = jt.code(output_shapes,output_types,inputs,cuda_header=CUDA_HEADER,cuda_src=CUDA_SRC,cuda_grad_src=CUDA_GRAD_SRC)
    return output



class ROIPool(nn.Module):
    def __init__(self, output_size, spatial_scale):
        super(ROIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def execute(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ")"
        return tmpstr