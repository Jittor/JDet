import jittor as jt
from jittor import  nn
from jittor.misc import _pair
import math

HEADER = r'''
#undef out
#include<executor.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t deformable_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                               const int height, const int width, scalar_t h, scalar_t w)
{

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                        const int h, const int w, const int height, const int width)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                          const int height, const int width, const scalar_t *im_data,
                                          const int data_width, const int bp_dir)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n, const scalar_t *data_im, const scalar_t *data_offset,
                                             const int height, const int width, const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
                                             const int batch_size, const int num_channels, const int deformable_group,
                                             const int height_col, const int width_col,
                                             scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    //const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const scalar_t map_h = i * dilation_h + offset_h;
          //const scalar_t map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}
template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    scalar_t *grad_im,const int whole_size)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
                                                        2 * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n, const scalar_t *data_col,
                                                   const scalar_t *data_im, const scalar_t *data_offset,
                                                   const int channels, const int height, const int width,
                                                   const int kernel_h, const int kernel_w,
                                                   const int pad_h, const int pad_w,
                                                   const int stride_h, const int stride_w,
                                                   const int dilation_h, const int dilation_w,
                                                   const int channel_per_deformable_group,
                                                   const int batch_size, const int offset_channels, const int deformable_group,
                                                   const int height_col, const int width_col, scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group *
                                                  batch_size * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
                                                channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 *
                                                        kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

'''
def deformable_im2col( data_im, data_offset, channels, height,width, ksize_h, ksize_w,
    pad_h,  pad_w,  stride_h, stride_w,
    dilation_h, dilation_w, parallel_imgs,
    deformable_group, columns_shape):

    src = f"""const int channels = {channels};
    const int height = {height};
    const int width = {width};
    const int ksize_h = {ksize_h};
    const int ksize_w = {ksize_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int stride_h = {stride_h}; 
    const int stride_w ={stride_w} ;
    const int dilation_h = {dilation_h}; 
    const int dilation_w = {dilation_w};
    const int parallel_imgs = {parallel_imgs}; 
    const int deformable_group = {deformable_group};
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col * parallel_imgs;
    int channel_per_deformable_group = channels / deformable_group; 
    cudaMemsetAsync(out0_p,0,out0->size);
    deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, in0_p, in1_p, height, width, ksize_h, ksize_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, channels, deformable_group,
            height_col, width_col, out0_p);

    """
    return jt.code(columns_shape,data_im.dtype,inputs=[data_im,data_offset],cuda_header = HEADER,cuda_src = src)

def deformable_col2im_coord(
    data_col, data_im, data_offset,
    channels, height, width, ksize_h,
    ksize_w, pad_h, pad_w,stride_h,
    stride_w, dilation_h, dilation_w,
    parallel_imgs, deformable_group, grad_offset_shape):
    src = f"""const int channels = {channels};
    const int height = {height};
    const int width = {width};
    const int ksize_h = {ksize_h};
    const int ksize_w = {ksize_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int stride_h = {stride_h}; 
    const int stride_w ={stride_w} ;
    const int dilation_h = {dilation_h}; 
    const int dilation_w = {dilation_w};
    const int parallel_imgs = {parallel_imgs}; 
    const int deformable_group = {deformable_group};
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group * parallel_imgs;
    int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

    cudaMemsetAsync(out0_p,0,out0->size);
    deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
    num_kernels, in0_p, in1_p, in2_p, channels, height, width,
    ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
    dilation_h, dilation_w, channel_per_deformable_group,
    parallel_imgs, 2 * ksize_h * ksize_w * deformable_group, deformable_group,
    height_col, width_col, out0_p);
    """
    return jt.code(grad_offset_shape,data_offset.dtype,[data_col, data_im, data_offset],cuda_header=HEADER,cuda_src=src)

 
def deformable_col2im(
    data_col, data_offset, channels,
    height, width, ksize_h,
    ksize_w, pad_h, pad_w,
    stride_h, stride_w,
    dilation_h, dilation_w,
    parallel_imgs, deformable_group,
    grad_im_shape):
    src = f"""const int channels = {channels};
    const int height = {height};
    const int width = {width};
    const int ksize_h = {ksize_h};
    const int ksize_w = {ksize_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int stride_h = {stride_h}; 
    const int stride_w ={stride_w} ;
    const int dilation_h = {dilation_h}; 
    const int dilation_w = {dilation_w};
    const int parallel_imgs = {parallel_imgs}; 
    const int deformable_group = {deformable_group};
    const int whole_size = {sum(grad_im_shape)};
    int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
    int channel_per_deformable_group = channels / deformable_group;

    cudaMemsetAsync(out0_p,0,out0->size);
    deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, in0_p, in1_p, channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, deformable_group, height_col, width_col, out0_p,whole_size);
    """    
    return jt.code(grad_im_shape,data_col.dtype,[data_col, data_offset],cuda_header=HEADER,cuda_src=src)

def deform_conv_forward_cuda(input,weight,offset,
                             kW, kH, dW, dH, padW, padH,
                             dilationW, dilationH, group,
                             deformable_group, im2col_step):

    batchSize,nInputPlane,inputHeight,inputWidth = input.shape

    nOutputPlane = weight.size(0)

    outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) // dW + 1
    outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) // dH + 1

    assert (offset.size(0) == batchSize), "invalid batch size of offset"


    columns_shape = (nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth)

    input = input.view((batchSize // im2col_step, im2col_step, nInputPlane,inputHeight, inputWidth))
    offset = offset.view((batchSize // im2col_step, im2col_step,deformable_group * 2 * kH * kW, outputHeight, outputWidth))

    output_buffer = jt.zeros((batchSize // im2col_step, nOutputPlane,
                 im2col_step * outputHeight, outputWidth),
                input.dtype)

    output_buffer = output_buffer.view((output_buffer.size(0), group, output_buffer.size(1) // group,output_buffer.size(2), output_buffer.size(3)))

    for elt in range(batchSize // im2col_step):
        columns = deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns_shape)

        columns = columns.view((group, columns.size(0) // group, columns.size(1)))
        weight = weight.view((group, weight.size(0) // group, weight.size(1),weight.size(2), weight.size(3)))
        
        for g in range(group):
            output_buffer[elt,g] = (output_buffer[elt,g].flatten(1)+jt.matmul(weight[g].flatten(1), columns[g])).view_as(output_buffer[elt,g])
        
    output_buffer = output_buffer.view((output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),output_buffer.size(3), output_buffer.size(4)))
    output_buffer = output_buffer.view((batchSize // im2col_step, nOutputPlane,im2col_step, outputHeight, outputWidth))
    output_buffer = output_buffer.transpose(0,2, 1,3,4)
    output = output_buffer.view((batchSize, nOutputPlane, outputHeight, outputWidth))

    return output 


def deform_conv_backward_input_cuda(input, offset,
                                    gradOutput,
                                     weight,
                                     kW, kH,dW,
                                    dH, padW,padH,dilationW,
                                    dilationH, group,
                                    deformable_group,im2col_step):

    batchSize,nInputPlane,inputHeight,inputWidth = input.shape
    nOutputPlane = weight.size(0)
    outputWidth =(inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) // dW + 1
    outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) // dH + 1

    assert offset.size(0) == batchSize, "invalid batch size of offset"

    gradInput = jt.zeros_like(input)
    gradOffset = jt.zeros_like(offset)
    columns = jt.zeros((nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth),input.dtype)

    gradOutput = gradOutput.view((batchSize // im2col_step, im2col_step,nOutputPlane, outputHeight, outputWidth))
    gradOutput = gradOutput.transpose(0,2,1,3,4)

    gradInput = gradInput.view((batchSize // im2col_step, im2col_step, nInputPlane,inputHeight, inputWidth))
    input = input.view((batchSize // im2col_step, im2col_step, nInputPlane,inputHeight, inputWidth))
    gradOffset = gradOffset.view((batchSize // im2col_step, im2col_step,deformable_group * 2 * kH * kW, outputHeight,outputWidth))
    offset = offset.view((batchSize // im2col_step, im2col_step,deformable_group * 2 * kH * kW, outputHeight, outputWidth))
    
    for elt in range(batchSize // im2col_step):
        columns = columns.view((group, columns.size(0) // group, columns.size(1)))
        weight = weight.view((group, weight.size(0) // group, weight.size(1),weight.size(2), weight.size(3)))
        gradOutput = gradOutput.view((gradOutput.size(0), group, gradOutput.size(1) // group, gradOutput.size(2), gradOutput.size(3), gradOutput.size(4)))
        
        for g in range(group):
            columns[g] = jt.matmul(weight[g].flatten(1).transpose(1, 0),gradOutput[elt,g].flatten(1))
        
        columns = columns.view((columns.size(0) * columns.size(1), columns.size(2)))
        gradOutput = gradOutput.view((gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2),gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)))
        
        gradOffset[elt] = deformable_col2im_coord(columns, input[elt], offset[elt], nInputPlane,
                                inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
                                dilationH, dilationW, im2col_step, deformable_group,
                                gradOffset[elt].shape)

        gradInput[elt] = deformable_col2im(columns, offset[elt], nInputPlane, inputHeight,inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                         dilationW, im2col_step, deformable_group, gradInput[elt].shape)

    gradInput = gradInput.view(batchSize, nInputPlane, inputHeight, inputWidth)
    gradOffset = gradOffset.view((batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth))
    return gradInput,gradOffset

def deform_conv_backward_parameters_cuda(
     input,  offset,  gradOutput,gradWeight,
      kW, kH, dW, dH,
    padW, padH, dilationW, dilationH, group,
    deformable_group, scale, im2col_step):
 
    batchSize,nInputPlane,inputHeight,inputWidth = input.shape
    nOutputPlane = gradWeight.size(0)
    outputWidth =(inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) // dW + 1
    outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) // dH + 1

    assert offset.size(0) == batchSize, "invalid batch size of offset"

    columns_shape=(nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth)

    gradOutput = gradOutput.view(batchSize // im2col_step, im2col_step,nOutputPlane, outputHeight, outputWidth)
    gradOutput = gradOutput.transpose(0,2,1,3,4)

    # gradOutputBuffer = jt.zeros_like(gradOutput)
    # gradOutputBuffer = gradOutputBuffer.view(batchSize // im2col_step, nOutputPlane, im2col_step,outputHeight, outputWidth)
    gradOutputBuffer = gradOutput
    gradOutputBuffer = gradOutputBuffer.view(batchSize // im2col_step, nOutputPlane,im2col_step * outputHeight, outputWidth)
    
    # gradOutput = gradOutput.transpose(0,2,1,3,4)
    # gradOutput = gradOutput.view(batchSize, nOutputPlane, outputHeight, outputWidth)

    input = input.view(batchSize // im2col_step, im2col_step, nInputPlane,inputHeight, inputWidth)
    offset = offset.view(batchSize // im2col_step, im2col_step, deformable_group * 2 * kH * kW, outputHeight, outputWidth)
    
    for elt in range( batchSize // im2col_step):
        columns = deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns_shape)

        gradOutputBuffer = gradOutputBuffer.view(gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) // group,
            gradOutputBuffer.size(2), gradOutputBuffer.size(3))
        columns = columns.view(group, columns.size(0) // group, columns.size(1))
        gradWeight = gradWeight.view(group, gradWeight.size(0) // group, gradWeight.size(1),gradWeight.size(2), gradWeight.size(3))
        
        for g in range(group):
            gradWeight[g] = (gradWeight[g].flatten(1)+scale*jt.matmul(gradOutputBuffer[elt,g].flatten(1),columns[g].transpose(1, 0))).view_as(gradWeight[g])

        gradOutputBuffer = gradOutputBuffer.view(gradOutputBuffer.size(0),
            gradOutputBuffer.size(1) * gradOutputBuffer.size(2),
            gradOutputBuffer.size(3), gradOutputBuffer.size(4))
        columns = columns.view(columns.size(0) * columns.size(1), columns.size(2))
        gradWeight = gradWeight.view(gradWeight.size(0) * gradWeight.size(1),
                                      gradWeight.size(2), gradWeight.size(3),
                                      gradWeight.size(4))
    return gradWeight


class DeformConvFunction(jt.Function):

    def execute(self,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                im2col_step=64):
        if input is not None and input.ndim != 4:
            raise ValueError(
                "Expected 4D tensor as input, got {}D tensor instead.".format(
                    input.ndim))
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step

        self.saved_tensors = (input, offset, weight)

        output_shape = DeformConvFunction._output_size(input, weight, self.padding,
                                            self.dilation, self.stride)


        if not jt.flags.use_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert (input.shape[0] %
                    cur_im2col_step) == 0, 'im2col step must divide batchsize'
            output = deform_conv_forward_cuda(
                input, weight, offset, weight.size(3), weight.size(2), self.stride[1], self.stride[0],
                self.padding[1], self.padding[0], self.dilation[1],
                self.dilation[0], self.groups, self.deformable_groups,
                cur_im2col_step)
            assert output.shape==output_shape
        return output


    def grad(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not jt.flags.use_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(self.im2col_step, input.shape[0])
            assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'

            grad_input,grad_offset = deform_conv_backward_input_cuda(
                input, offset, grad_output, weight,weight.size(3),
                weight.size(2), self.stride[1], self.stride[0],
                self.padding[1], self.padding[0], self.dilation[1],
                self.dilation[0], self.groups, self.deformable_groups,
                cur_im2col_step)
            
            # print(grad_output.sum(),grad_input.sum(),grad_offset.sum())

            grad_weight = jt.zeros_like(weight)
            grad_weight = deform_conv_backward_parameters_cuda(
                input, offset, grad_output,
                grad_weight, weight.size(3),
                weight.size(2), self.stride[1], self.stride[0],
                self.padding[1], self.padding[0], self.dilation[1],
                self.dilation[0], self.groups, self.deformable_groups, 1,
                cur_im2col_step)

        return grad_input, grad_offset, grad_weight

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.ndim - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

deform_conv = DeformConvFunction.apply

class DeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = jt.zeros((out_channels, in_channels // self.groups,*self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        nn.init.uniform_(self.weight,-stdv, stdv)

    def execute(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups, self.deformable_groups)
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        # input_pad = (
        #     x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        # if input_pad:
        #     pad_h = max(self.kernel_size[0] - x.size(2), 0)
        #     pad_w = max(self.kernel_size[1] - x.size(3), 0)
        #     x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        #     offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant',
        #                    0).contiguous()
        # out = deform_conv(x, offset, self.weight, self.stride, self.padding,
        #                   self.dilation, self.groups, self.deformable_groups)
        # if input_pad:
        #     out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
        #               pad_w].contiguous()
        # return out


def test_dcn():
    import numpy as np
    dcn = DeformConv(256, 256, (3, 3), (1, 1), padding=(1, 1), dilation=(1, 1), groups=1, deformable_groups=1)
    dcn.load_state_dict(jt.load("/home/lxl/workspace/s2anet/dcn_init.pth"))
    # x,offset = jt.load("/home/lxl/workspace/JDet/test_dcn1.pkl")
    x,offset,out = jt.load("/home/lxl/workspace/JDet/test_dcn_final.pkl")
    x = jt.array(x)
    offset = jt.array(offset)

    output = dcn(x,offset )
    # print(out.sum())
    # print(np.abs(output.numpy()-out).max())

    loss = output.sum()
    # print(loss)
    print(output.sum())
    grads = jt.grad(loss,[x,offset,dcn.weight])
    print(grads[0].sum())
    print(grads[1].sum())
    print(grads[2].sum())


if __name__ == "__main__":
    jt.flags.use_cuda=1
    test_dcn()