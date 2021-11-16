from jittor import nn
import jittor as jt
from jittor.misc import _pair

__all__ = ["ROIAlign"]

CUDA_HEADER = r'''
#include <cmath>
#include <cstdio>
#include <climits>
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y < 0) y = 0;
  if (x < 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename scalar_t>
__global__ void ROIAlignRotatedForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale - (scalar_t)0.5;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale - (scalar_t)0.5;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // Force malformed ROIs to be 1x1 
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    const scalar_t* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
        ? sample_num
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);
    
    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosscalar_theta = cos(theta);
    scalar_t sinscalar_theta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1);  // e.g. = 4

    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
        const scalar_t yy = roi_start_h + ph * bin_size_h +
            static_cast<scalar_t>(iy + .5f) * bin_size_h /
                static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        scalar_t x = xx * cosscalar_theta + yy * sinscalar_theta + roi_center_w;
        scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        // scalar_t x = xx * cosscalar_theta - yy * sinscalar_theta + roi_center_w;
        // scalar_t y = xx * sinscalar_theta + yy * cosscalar_theta + roi_center_h;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x);
        output_val += val;
        }
    }
    output_val /= count;

    top_data[index] = output_val;
    }
}
template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y < 0) y = 0;
  if (x < 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void ROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale - (scalar_t)0.5;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale - (scalar_t)0.5;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    scalar_t* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const scalar_t* offset_top_diff = top_diff + top_offset;
    const scalar_t top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sample_num > 0)
        ? sample_num
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy = roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        scalar_t x = xx * cosTheta + yy * sinTheta + roi_center_w;
        scalar_t y = yy * cosTheta - xx * sinTheta + roi_center_h;
        // scalar_t x = xx * cosTheta - yy * sinTheta + roi_center_w;
        // scalar_t y = xx * sinTheta + yy * cosTheta + roi_center_h;

        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high);

        scalar_t g1 = top_diff_this_bin * w1 / count;
        scalar_t g2 = top_diff_this_bin * w2 / count;
        scalar_t g3 = top_diff_this_bin * w3 / count;
        scalar_t g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, g4);
        }  // if
      }  // ix
    }  // iy
  }  // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward
'''
class _RotatedROIAlign_v1(jt.Function):
    def execute(self,input, rois, output_size, spatial_scale, sampling_ratio):
        self.input = input 
        self.rois = rois 
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        assert rois.shape[1]==6
        output_shapes = (rois.shape[0], input.shape[1], output_size[0], output_size[1])
        return jt.code(output_shapes,input.dtype,[input,rois],cuda_header=CUDA_HEADER,
                          cuda_src=f'''
                            @alias(input,in0);
                            @alias(rois,in1);
                            @alias(output,out0);
                            const float spatial_scale = {spatial_scale};
                            const float  sampling_ratio = {sampling_ratio};
                            auto num_rois = rois_shape0;
                            auto channels = input_shape1;
                            auto height = input_shape2;
                            auto width = input_shape3;
                            auto pooled_height = output_shape2;
                            auto pooled_width = output_shape3;
                            auto output_size = num_rois * pooled_height * pooled_width * channels;
                            ROIAlignRotatedForward<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>> (
                                        output_size, input_p, rois_p, spatial_scale,
                                        sampling_ratio, channels, height, width, pooled_height,
                                        pooled_width, output_p);
                            ''')
    def grad(self,output_grad):
        input,rois = self.input,self.rois
        input_grad =  jt.code(input.shape,input.dtype,[input,rois,output_grad],
                      cuda_header=CUDA_HEADER,
                      cuda_src=f'''
                            @alias(input,in0)
                            @alias(rois,in1)
                            @alias(grad,in2)
                            @alias(grad_input,out0)
                            const float spatial_scale = {self.spatial_scale};
                            const float  sampling_ratio = {self.sampling_ratio};
                            auto num_rois = rois_shape0;
                            auto channels = input_shape1;
                            auto height = input_shape2;
                            auto width = input_shape3;
                            auto pooled_height = grad_shape2;
                            auto pooled_width = grad_shape3;
                            auto output_size = num_rois * pooled_height * pooled_width * channels;
                            cudaMemsetAsync(grad_input_p,0,grad_input->size);
                             ROIAlignBackward<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                                    output_size, grad_p, rois_p, spatial_scale, sampling_ratio,
                                    channels, height, width, pooled_height, pooled_width,
                                    grad_input_p);
                            ''')
        return input_grad,None

roi_align = _RotatedROIAlign_v1.apply

class ROIAlignRotated_v1(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio=0):
        super(ROIAlignRotated_v1, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def execute(self, input, rois):
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr


def test_roialign():
    jt.flags.use_cuda=1
    roi_align = ROIAlignRotated_v1((7,7),1/16.)
    feature = jt.randn((2,1024,64,64))
    roi = jt.array([[0,20,120,80,195.5,0.3],[1,23,56,200,300.5,0.2]])
    output = roi_align(feature,roi)
    print(output.shape,output.sum())
    print(jt.grad(output.exp(),feature))
  
if __name__ == "__main__":
    test_roialign()