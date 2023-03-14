import jittor as jt
from jittor import nn
from jittor.misc import _pair

riroi_cuda_head = r'''
#include <math.h>
#include <stdio.h>
#define PI 3.141592653

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

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

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
__global__ void RiROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const int nOrientation,
                                scalar_t *top_data) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);
    
    // TODO
    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * PI);
    int ind =  floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation; 
    
    const scalar_t* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) * height * width;

    const scalar_t* offset_bottom_data_plus =
        bottom_data + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot_plus) * height * width;

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
    const scalar_t count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

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
        // scalar_t x = xx * cosscalar_theta + yy * sinscalar_theta + roi_center_w;
        // scalar_t y = yy * cosscalar_theta - xx * sinscalar_theta + roi_center_h;
        scalar_t x = xx * cosscalar_theta - yy * sinscalar_theta + roi_center_w;
        scalar_t y = xx * sinscalar_theta + yy * cosscalar_theta + roi_center_h;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x);
        scalar_t val_plus = bilinear_interpolate<scalar_t>(
            offset_bottom_data_plus, height, width, y, x);
        output_val += r_var * val + l_var * val_plus;
        }
    }
    output_val /= count;
    top_data[index] = output_val;
    }
}

template <typename scalar_t>
int RiROIAlignForwardLaucher(const scalar_t *bottom_data, const scalar_t *rois_data,
                                const float spatial_scale, const int sample_num,
                                const int channels, const int height,
                                const int width, const int num_rois,
                                const int pooled_height, const int pooled_width,
                                const int nOrientation,
                                scalar_t *top_data) {
    const int output_size = num_rois * pooled_height * pooled_width * channels * nOrientation;
    RiROIAlignForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>> (
            output_size, bottom_data, rois_data, scalar_t(spatial_scale),
            sample_num, channels, height, width, pooled_height,
            pooled_width, nOrientation, top_data);
    return 1;
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

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

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
__global__ void RiROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int nOrientation, scalar_t *bottom_diff) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int o = (index / pooled_width / pooled_height) % nOrientation;
    int c = (index / pooled_width / pooled_height / nOrientation) % channels;
    int n = index / pooled_width / pooled_height / nOrientation / channels;

    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not round
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // scalar_t theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];
    

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    // find aligned index
    scalar_t ind_float = theta * nOrientation / (2 * PI);
    int ind =  floor(ind_float);
    scalar_t l_var = ind_float - (scalar_t)ind;
    scalar_t r_var = 1.0 - l_var;
    // correct start channel
    ind = (ind + nOrientation) % nOrientation;
    // rotated channel
    int ind_rot = (o - ind + nOrientation) % nOrientation;
    int ind_rot_plus = (ind_rot + 1 + nOrientation) % nOrientation; 
   
    scalar_t* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot) * height * width;

    scalar_t* offset_bottom_diff_plus =
        bottom_diff + (roi_batch_ind * channels * nOrientation + c * nOrientation + ind_rot_plus) * height * width;


    int top_offset = (n * channels * nOrientation + c * nOrientation + o) * pooled_height * pooled_width;
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
        // scalar_t x = xx * cosTheta + yy * sinTheta + roi_center_w;
        // T y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = xx * cosTheta - yy * sinTheta + roi_center_w;
        scalar_t y = xx * sinTheta + yy * cosTheta + roi_center_h;

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
              offset_bottom_diff + y_low * width + x_low, g1*r_var);
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, g2*r_var);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, g3*r_var);
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, g4*r_var);
          
          atomicAdd(
              offset_bottom_diff_plus + y_low * width + x_low, g1*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_low * width + x_high, g2*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_high * width + x_low, g3*l_var);
          atomicAdd(
              offset_bottom_diff_plus + y_high * width + x_high, g4*l_var);
        }  // if
      }  // ix
    }  // iy
  }  // CUDA_1D_KERNEL_LOOP
}  // RoIAlignBackward

template <typename scalar_t>
int RiROIAlignBackwardLaucher(const scalar_t *top_diff, const scalar_t *rois_data,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            const int nOrientation,
                            scalar_t *bottom_diff) {
    const int output_size = num_rois * pooled_height * pooled_width * channels * nOrientation;
    if (sizeof(scalar_t) == sizeof(double)) {
        fprintf(stderr, "double is not supported\n");
        exit(-1);
    }

    RiROIAlignBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
            output_size, top_diff, rois_data, spatial_scale, sample_num,
            channels, height, width, pooled_height, pooled_width, nOrientation,
            bottom_diff);
    return 1;
}
'''

class _RiRoIAlign(jt.Function):
    def execute(self, features, rois, out_size, spatial_scale, sample_num=0, nOrientation=8):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num
        self.rois = rois
        self.feature_size = features.shape
        self.nOrientation = nOrientation
        batch_size, num_channels, data_height, data_width = features.shape
        num_rois = rois.shape[0]
        
        # TODO: check output not inplace
        if jt.flags.use_cuda:
          riroi_forward_cuda_src = f'''
            @alias(features,in0);
            @alias(rois, in1);
            @alias(output, out0);
            const float spatial_scale = {spatial_scale};
            const int pooled_height = {out_h};
            const int pooled_width = {out_w};
            const int sample_num = {sample_num};
            const int nOrientation = {nOrientation};
            const int num_rois = rois_shape0;
            const int size_rois = rois_shape1;
            cudaMemsetAsync(output_p,0,output->size);
            const int num_channels = features_shape1 / nOrientation;
            const int data_height = features_shape2;
            const int data_width = features_shape3;
            RiROIAlignForwardLaucher(features_p, rois_p, spatial_scale, sample_num,
                                  num_channels, data_height, data_width, num_rois,
                                  pooled_height, pooled_width, nOrientation, output_p);
          '''
          output = jt.code(shape=(num_rois, num_channels, out_h, out_w),dtype=features.dtype,
                  inputs=[features, rois],
                  cuda_header=riroi_cuda_head, cuda_src=riroi_forward_cuda_src)
          return output
        else:
            raise NotImplementedError

    def grad(self, grad_output):
        feature_size = self.feature_size
        spatial_scale = self.spatial_scale
        rois = self.rois
        sample_num = self.sample_num
        nOrientation = self.nOrientation
        assert (feature_size is not None and jt.flags.use_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.shape[3]
        out_h = grad_output.shape[2]

        grad_rois = None
        riroi_backword_cuda_src = f'''
          @alias(top_grad, in0);
          @alias(rois, in1);
          @alias(bottomgrad, out0);
          const float spatial_scale = {spatial_scale};
          const int pooled_height = {out_h};
          const int pooled_width = {out_w};
          const int sample_num = {sample_num};
          const int nOrientation = {nOrientation};
          const int num_rois = rois_shape0;
          const int size_rois = rois_shape1;
          cudaMemsetAsync(bottomgrad_p,0,bottomgrad->size);
          
          const int num_channels = bottomgrad_shape1 / nOrientation;
          const int data_height = bottomgrad_shape2;
          const int data_width = bottomgrad_shape3;

          RiROIAlignBackwardLaucher(top_grad_p, rois_p, spatial_scale, sample_num,
                                  num_channels, data_height, data_width, num_rois,
                                  pooled_height, pooled_width, nOrientation, bottomgrad_p);
        '''
        grad_input = jt.code(shape=(batch_size, num_channels, data_height, data_width), dtype=rois.dtype,
                inputs=[grad_output, rois],
                cuda_header=riroi_cuda_head, cuda_src=riroi_backword_cuda_src)
        return grad_input, grad_rois, None, None, None, None

riroi_align = _RiRoIAlign.apply

class RiRoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sample_num=0, nOrientation=8):
        super(RiRoIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.nOrientation = int(nOrientation)

    def execute(self, features, rois):
        return _RiRoIAlign.apply(features, rois, self.output_size,
                                  self.spatial_scale, self.sample_num, self.nOrientation)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sample_num=" + str(self.sample_num)
        tmpstr += ", nOrientation=" + str(self.nOrientation)
        tmpstr += ")"
        return tmpstr
