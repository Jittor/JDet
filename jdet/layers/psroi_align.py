from jittor import nn
import jittor as jt
from jittor.misc import _pair

__all__ = ["PSROIAlign"]

CUDA_HEADER = r'''
#include <cmath>
#include <cstdio>
#include <climits>
#define CUDA_1D_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
using namespace std;
__device__ float bilinear_interpolate(const float* bottom_data,
    const int height, const int width,
    float y, float x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float) y_low;
  } else {
    y_high = y_low + 1;
  }
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float) x_low;
  } else {
    x_high = x_low + 1;
  }
  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  float v1 = bottom_data[y_low * width + x_low];
  float v2 = bottom_data[y_low * width + x_high];
  float v3 = bottom_data[y_high * width + x_low];
  float v4 = bottom_data[y_high * width + x_high];
  float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    float y, float x,
    float & w1, float & w2, float & w3, float & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }
  if (y <= 0) y = 0;
  if (x <= 0) x = 0;
  y_low = (int) y;
  x_low = (int) x;
  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (float) y_low;
  } else {
    y_high = y_low + 1;
  }
  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (float) x_low;
  } else {
    x_high = x_low + 1;
  }
  float ly = y - y_low;
  float lx = x - x_low;
  float hy = 1. - ly, hx = 1. - lx;
  // reference in forward
  // float v1 = bottom_data[y_low * width + x_low];
  // float v2 = bottom_data[y_low * width + x_high];
  // float v3 = bottom_data[y_high * width + x_low];
  // float v4 = bottom_data[y_high * width + x_high];
  // float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  return;
}
'''

CUDA_SRC = r'''
__global__ void PSRoIAlignForward(@ARGS_DEF,const int nthreads, const float* bottom_data,
    const int channels,const int height, const int width,const int aligned_height, const int aligned_width,
    const float* bottom_rois, float* top_data) {
            @PRECALC
    // printf("%d\n", out_dim);
    const float spatial_scale = @in2(0);
    const float sampling_ratio = @in2(1);
    // const int out_dim = @in2(2);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // NOfloatE Mingtao: copied from light-head
        // the order of dims is changed because the original implementation is in floatensorFlow
        int n = index;
        int pw = n % aligned_width;
        n /= aligned_width;
        int ph = n % aligned_height;
        n /= aligned_height;
        int ctop = n % out_dim;
        n /= out_dim;
        const float* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        // Do not using rounding; this implementation detail is critical
        auto roi_start_w = offset_bottom_rois[1] * spatial_scale;
        auto roi_start_h = offset_bottom_rois[2] * spatial_scale;
        auto roi_end_w = offset_bottom_rois[3] * spatial_scale;
        auto roi_end_h = offset_bottom_rois[4] * spatial_scale;
        // Force malformed ROIs to be 1x1
        auto roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
        auto roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
        auto bin_size_h = roi_height / aligned_height;
        auto bin_size_w = roi_width / aligned_width;
        // **************************************************************************************
        int c = ( ctop * aligned_width + ph ) * aligned_width + pw;
        // **************************************************************************************
        const float* offset_bottom_data =
            bottom_data + (roi_batch_ind * channels + c) * height * width; //channels?
        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / aligned_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);
        // We do average (integral) pooling inside a bin
        const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
        float output_val = 0.;
        for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
        {
            const float y = roi_start_h + ph * bin_size_h +
                (iy + .5f) * bin_size_h / roi_bin_grid_h;  // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const float x = roi_start_w + pw * bin_size_w +
                    (ix + .5f) * bin_size_w / roi_bin_grid_w;
                float val = bilinear_interpolate(
                        offset_bottom_data, height, width, y, x, index);
                output_val += val;
            }
        }
        output_val /= count;
        top_data[index] = output_val;
    }
}
@alias(input,in0);
@alias(rois,in1);
@alias(output,out0);
auto num_rois = rois_shape0;
auto channels = input_shape1;
auto height = input_shape2;
auto width = input_shape3;
auto aligned_height = output_shape2;
auto aligned_width = output_shape3;
auto output_size = num_rois * aligned_height * aligned_width * out_dim;
const int total_count = in1_shape0 * out0_shape2 * out0_shape3 * in0_shape1;
const int thread_per_block = 512L;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
PSRoIAlignForward<<<block_count, thread_per_block>>>(@ARGS,output_size,input_p,channels,
height, width,aligned_height,aligned_width,rois_p,output_p);
'''

CUDA_GRAD_SRC = [r'''
__global__ void PSRoIAlignBackwardFeature(@ARGS_DEF,const int nthreads, const float* top_diff,
    const int num_rois,
    const int channels, const int height, const int width,
    const int aligned_height, const int aligned_width,
    float* bottom_diff,
    const float* bottom_rois) {
         @PRECALC
         
@alias(input,in0)
@alias(rois,in1)
@alias(grad_input,out0)
@alias(grad,dout)
    const float spatial_scale = @in2(0);
    const float sampling_ratio = @in2(1);
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // NOfloatE Mingtao: copied from light-head
        // the order of dims is changed because the original implementation is in floatensorFlow
        int n = index;
        int pw = n % aligned_width;
        n /= aligned_width;
        int ph = n % aligned_height;
        n /= aligned_height;
        int ctop = n % out_dim;
        n /= out_dim;
        const float* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        // Do not using rounding; this implementation detail is critical
        float roi_start_w = offset_bottom_rois[1] * spatial_scale;
        float roi_start_h = offset_bottom_rois[2] * spatial_scale;
        float roi_end_w = offset_bottom_rois[3] * spatial_scale;
        float roi_end_h = offset_bottom_rois[4] * spatial_scale;
        // Force malformed ROIs to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
        float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
        float bin_size_h = roi_height / aligned_height;
        float bin_size_w = roi_width / aligned_width;
        int c = ( ctop * aligned_width + ph ) * aligned_width + pw;
        float* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
        
        int top_offset = (n * out_dim + ctop) * aligned_height * aligned_width; 
        const float* offset_top_diff = top_diff + top_offset;
        const float top_diff_this_bin = offset_top_diff[ph * aligned_width + pw];
        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
            ? sampling_ratio
            : ceil(roi_height / aligned_height); // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);
        // We do average (integral) pooling inside a bin
        const float count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4
        for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
        {
            const float y = roi_start_h + ph * bin_size_h +
                (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                const float x = roi_start_w + pw * bin_size_w +
                    (ix + .5f) * bin_size_w / roi_bin_grid_w;
                float w1, w2, w3, w4;
                int x_low, x_high, y_low, y_high;
                bilinear_interpolate_gradient(
                        height, width, y, x, w1, w2, w3, w4,
                        x_low, x_high, y_low, y_high, index);
                float g1 = top_diff_this_bin * w1 / count;
                float g2 = top_diff_this_bin * w2 / count;
                float g3 = top_diff_this_bin * w3 / count;
                float g4 = top_diff_this_bin * w4 / count;
                if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                    atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<float>(g1));
                    atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<float>(g2));
                    atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<float>(g3));
                    atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<float>(g4));
                } // if
            } // ix
        } // iy
    } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward
auto num_rois = rois_shape0;
auto channels = input_shape1;
auto height = input_shape2;
auto width = input_shape3;
auto aligned_height = grad_shape2;
auto aligned_width = grad_shape3;
auto output_size = num_rois * aligned_height * aligned_width * out_dim;
cudaMemsetAsync(grad_input_p,0,grad_input->size);
const int total_count = rois_shape0 * grad_shape2 * grad_shape3 * input_shape1;
const int thread_per_block = 512;
const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
PSRoIAlignBackwardFeature<<<block_count, thread_per_block>>>(@ARGS,output_size,grad_p,num_rois,
channels, height, width, aligned_height, aligned_width, grad_input_p, rois_p);
''','','']

def psroi_align(input, rois, output_size, spatial_scale, sampling_ratio, out_dim):
    output_size = _pair(output_size)
    options = jt.array([spatial_scale,sampling_ratio, out_dim])
    output_shapes = (rois.shape[0], input.shape[1], output_size[0], output_size[1])
    inputs = [input,rois,options]
    output_types = input.dtype
    if rois.shape[0]==0:
      return jt.zeros(output_shapes,input.dtype)
    output = jt.code(output_shapes,output_types,inputs,cuda_header=CUDA_HEADER,cuda_src=CUDA_SRC,cuda_grad_src=CUDA_GRAD_SRC)
    return output


class PSROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, out_dim):
        super(PSROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.out_dim = out_dim

    def execute(self, input, rois):
        return psroi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.out_dim
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr


def test_psroi_align():
    jt.flags.use_cuda=1
    psroialign = PSROIAlign(7,1.0,sampling_ratio=0,out_dim=23)

    feature = jt.rand((8,512,64,64),dtype="float32")
    boxes = jt.array([[0,1.2,10,3,10],[3,1,67,2,34]])
    output = psroialign(feature,boxes)
    print(output.shape)
    # print(jt.grad(output,feature))


if __name__ == "__main__":
    test_psroi_align()