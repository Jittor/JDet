#coding=utf-8
import jittor as jt 
from jittor import nn
from jittor.misc import _pair 
import numpy as np
import math
from jdet.utils.registry import HEADS

__all__ = ['DCN']

def dcn_v2_conv_forward(input,offset,mask,weight,bias,stride,padding,dilation,deformable_groups):
    kernel_size = weight.shape[2:4]
    batch = input.shape[0]
    channels = input.shape[1]
    height = input.shape[2]
    width  = input.shape[3]
    channels_out = weight.shape[0]
    
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]


    height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ones = jt.ones((batch, height_out, width_out),dtype=input.dtype)
    colums = jt.empty((batch, channels * kernel_h * kernel_w, 1 * height_out * width_out),dtype=input.dtype)
    inputs = [input,weight,bias,offset,mask,ones,colums]

    output_shape = (batch, channels_out, height_out, width_out)
    output_type = input.dtype
    output = jt.code(output_shape,output_type,inputs,
    cuda_header=r'''
#undef out
#include<cstdio>
#include<cstring>
#include<algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <executor.h>
using namespace std;
namespace jittor {
extern cublasHandle_t cublas_handle;
} // jittor
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    const int c_col = c_im * kernel_h * kernel_w;
    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
__global__ void createBatchGemmBuffer(const float **input_b, float **output_b,
                                      float **columns_b, const float **ones_b,
                                      const float **weight_b, const float **bias_b,
                                      float *input, float *output,
                                      float *columns, float *ones,
                                      float *weight, float *bias,
                                      const int input_stride, const int output_stride,
                                      const int columns_stride, const int ones_stride,
                                      const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // share weights and bias within a Mini-Batch
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}
void modulated_deformable_im2col_cuda(
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
}
    ''',
    cuda_src=f'''
    const int kernel_h = {kernel_h};
    const int kernel_w = {kernel_w};
    const int stride_h = {stride_h};
    const int stride_w = {stride_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int dilation_h = {dilation_h};
    const int dilation_w = {dilation_w};
    const int deformable_group = {deformable_groups};
'''
+
r'''
     @alias(input,in0)
    @alias(weight,in1)
    @alias(bias,in2)
    @alias(offset,in3)
    @alias(mask,in4)
    @alias(ones,in5)
    @alias(columns,in6)
    @alias(output,out0)
    const int batch = input_shape0;
    const int channels = input_shape1;
    const int height = input_shape2;
    const int width = input_shape3;
    const int channels_out = weight_shape0;
    const int channels_kernel = weight_shape1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    // prepare for batch-wise computing, which is significantly faster than instance-wise computing
    // when batch size is large.
    // launch batch threads
    int matrices_size = batch * sizeof(float *);
    const float ** input_b;
    float ** output_b;
    float ** columns_b;
    const float ** ones_b;
    const float ** weight_b;
    const float ** bias_b;
    size_t input_b_allocation;
    size_t output_b_allocation;
    size_t columns_b_allocation;
    size_t ones_b_allocation;
    size_t weight_b_allocation;
    size_t bias_b_allocation;
    input_b = (const float **)exe.allocator->alloc(matrices_size, input_b_allocation);
    output_b = (float **)exe.allocator->alloc(matrices_size, output_b_allocation);
    columns_b = (float **)exe.allocator->alloc(matrices_size, columns_b_allocation);
    ones_b = (const float **)exe.allocator->alloc(matrices_size, ones_b_allocation);
    weight_b = (const float **)exe.allocator->alloc(matrices_size, weight_b_allocation);
    bias_b = (const float **)exe.allocator->alloc(matrices_size, bias_b_allocation);
    const int block = 128;
    const int grid = (batch + block - 1) / block;
    createBatchGemmBuffer<<<grid, block>>>(
        input_b, output_b,
        columns_b, ones_b,
        weight_b, bias_b,
        input_p,
        output_p,
        columns_p,
        ones_p,
        weight_p,
        bias_p,
        channels * width * height,
        channels_out * width_out * height_out,
        channels * kernel_h * kernel_w * height_out * width_out,
        height_out * width_out,
        batch);
    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;
    cublasHandle_t& handle = cublas_handle;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemmBatched(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            n_,
                            m_,
                            k_,
                            &alpha,
                            ones_b, k_,
                            bias_b, k_,
                            &beta,
                            output_b, n_,
                            batch);
    modulated_deformable_im2col_cuda(input_p,
                                     offset_p,
                                     mask_p,
                                     batch, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w,
                                     pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                     deformable_group,
                                     columns_p);
    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;
    float beta2 = 1.0f;
    cublasSgemmBatched(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            n,
                            m,
                            k,
                            &alpha,
                            (const float **)columns_b, n,
                            weight_b, k,
                            &beta2,
                            output_b, n,
                            batch);
    exe.allocator->free(input_b, matrices_size, input_b_allocation);
    exe.allocator->free(output_b, matrices_size, output_b_allocation);
    exe.allocator->free(columns_b, matrices_size, columns_b_allocation);
    exe.allocator->free(ones_b, matrices_size, ones_b_allocation);
    exe.allocator->free(weight_b, matrices_size, weight_b_allocation);
    exe.allocator->free(bias_b, matrices_size, bias_b_allocation);
''')
    return output

def dcn_v2_conv_backward(input,offset,mask,weight,bias,grad_output,stride,padding,dilation,deformable_groups):
    kernel_size = weight.shape[2:4]
    batch = input.shape[0]
    channels = input.shape[1]
    height = input.shape[2]
    width  = input.shape[3]
    channels_out = weight.shape[0]
    
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]
    dilation_h = dilation[0]
    dilation_w = dilation[1]


    height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ones = jt.ones((batch, height_out, width_out),dtype=input.dtype)
    colums = jt.empty((batch, channels * kernel_h * kernel_w, 1 * height_out * width_out),dtype=input.dtype)
    inputs = [input,weight,bias,offset,mask,ones,colums,grad_output]

    output_shape = [input.shape,weight.shape,bias.shape,offset.shape,mask.shape]
    output_type = [input.dtype,weight.dtype,bias.dtype,offset.dtype,mask.dtype]
    input_grad,weight_grad,bias_grad,offset_grad,mask_grad = jt.code(output_shape,output_type,inputs,cuda_header=r'''
#include<cstdio>
#include<cstring>
#include<algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;
namespace jittor {
extern cublasHandle_t cublas_handle;
} // jittor
#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
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
  float weight = 0;
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
__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
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
  float weight = 0;
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
__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    const int c_col = c_im * kernel_h * kernel_w;
    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}
__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset, const float *data_mask,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
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
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;
    const float cur_top_grad = data_col[index] * mask;
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
          float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}
__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *data_col, const float *data_im,
                                                             const float *data_offset, const float *data_mask,
                                                             const int channels, const int height, const int width,
                                                             const int kernel_h, const int kernel_w,
                                                             const int pad_h, const int pad_w,
                                                             const int stride_h, const int stride_w,
                                                             const int dilation_h, const int dilation_w,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels, const int deformable_group,
                                                             const int height_col, const int width_col,
                                                             float *grad_offset, float *grad_mask)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output
    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
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
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      else
      {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const float weight = dmcn_get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}
void modulated_deformable_im2col_cuda(
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
}
void modulated_deformable_col2im_cuda(
  const float* data_col, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group, float* grad_im){
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, height_col, width_col, grad_im);
}
void modulated_deformable_col2im_coord_cuda(
  const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group,
  float* grad_offset, float* grad_mask) {
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col, 
        grad_offset, grad_mask);
}
    ''',
    cuda_src=f'''
    const int kernel_h = {kernel_h};
    const int kernel_w = {kernel_w};
    const int stride_h = {stride_h};
    const int stride_w = {stride_w};
    const int pad_h = {pad_h};
    const int pad_w = {pad_w};
    const int dilation_h = {dilation_h};
    const int dilation_w = {dilation_w};
    const int deformable_group = {deformable_groups};
    '''+r'''
    @alias(input,in0)
    @alias(weight,in1)
    @alias(bias,in2)
    @alias(offset,in3)
    @alias(mask,in4)
    @alias(ones,in5)
    @alias(columns,in6)
    @alias(grad_output,in7)
    @alias(grad_input,out0)
    @alias(grad_weight,out1)
    @alias(grad_bias,out2)
    @alias(grad_offset,out3)
    @alias(grad_mask,out4)
    const int batch = input_shape0;
    const int channels = input_shape1;
    const int height = input_shape2;
    const int width = input_shape3;
    const int channels_out = weight_shape0;
    const int channels_kernel = weight_shape1;
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    for (int b = 0; b < batch; b++)
    {
        auto input_n = input_p+input_stride0*b;
        auto offset_n = offset_p+offset_stride0*b;
        auto mask_n = mask_p+mask_stride0*b;
        auto grad_output_n = grad_output_p+grad_output_stride0*b;
        auto grad_input_n = grad_input_p+grad_input_stride0*b;
        auto grad_offset_n = grad_offset_p+grad_offset_stride0*b;
        auto grad_mask_n = grad_mask_p+grad_mask_stride0*b;
        long m = channels * kernel_h * kernel_w;
        long n = height_out * width_out;
        long k = channels_out;
        float alpha0  = 1.0f;
        float beta0 = 0.0f;
        cublasHandle_t& handle = cublas_handle;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha0,
                         grad_output_n, n,
                         weight_p, m, &beta0,
                         columns_p, n);
        // gradient w.r.t. input coordinate data
        modulated_deformable_col2im_coord_cuda(columns_p,
                                               input_n,
                                               offset_n,
                                               mask_n,
                                               1, channels, height, width,
                                               height_out, width_out, kernel_h, kernel_w,
                                               pad_h, pad_w, stride_h, stride_w,
                                               dilation_h, dilation_w, deformable_group,
                                               grad_offset_n,
                                               grad_mask_n);
        // gradient w.r.t. input data
        modulated_deformable_col2im_cuda(columns_p,
                                         offset_n,
                                         mask_n,
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         grad_input_n);
        // gradient w.r.t. weight, dWeight should accumulate across the batch and group
        modulated_deformable_im2col_cuda(
                                         input_n,
                                         offset_n,
                                         mask_n,
                                         1, channels, height, width,
                                         height_out, width_out, kernel_h, kernel_w,
                                         pad_h, pad_w, stride_h, stride_w,
                                         dilation_h, dilation_w, deformable_group,
                                         columns_p);
        long m_ = channels_out;
        long n_ = channels * kernel_h * kernel_w;
        long k_ = height_out * width_out;
        float alpha  = 1.0f;
        float beta = 1.0f;
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_, m_, k_, &alpha,
                         columns_p, k_,
                         grad_output_n, k_, &beta,
                         grad_weight_p, n_);
        //cublasDestroy(handle);
        // gradient w.r.t. bias
        // long m_ = channels_out;
        // long k__ = height_out * width_out;
        cublasSgemv(handle,
                         CUBLAS_OP_T,
                         k_, m_, &alpha,
                         grad_output_n, k_,
                         ones_p, 1, &beta,
                         grad_bias_p, 1);
    }
    ''')
    return input_grad,offset_grad,mask_grad,weight_grad,bias_grad




class DCN_V2_CONV(jt.Function):

    def execute(self, input,offset,mask,weight,bias,stride,padding,dilation,deformable_groups):
        self.input = input
        self.offset = offset
        self.mask = mask 
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        output =  dcn_v2_conv_forward(input,offset,mask,weight,bias,stride,padding,dilation,deformable_groups)
        return output
            
    def grad(self, grad_output):
        input_grad,offset_grad,mask_grad,weight_grad,bias_grad = dcn_v2_conv_backward(self.input,self.offset,self.mask,self.weight,self.bias,grad_output,self.stride,self.padding,self.dilation,self.deformable_groups)
        return input_grad,offset_grad,mask_grad,weight_grad,bias_grad,None, None, None, None

dcn_v2_conv = DCN_V2_CONV.apply


def dcn_v2_pooling_forward(input, bbox, trans, spatial_scale,pooled_size,output_dim,no_trans,group_size,part_size,sample_per_part,trans_std):
    channels = input.shape[1]
    num_bbox = bbox.shape[0]
    assert channels == output_dim, "input channels and output channels must equal"
    pooled_height = pooled_size
    pooled_width = pooled_size
    output_shape = [(num_bbox, output_dim, pooled_height, pooled_width),(num_bbox, output_dim, pooled_height, pooled_width)]
    output_dtypes= [input.dtype,input.dtype]
    inputs = [input,bbox,trans]
    out,top_count = jt.code(output_shape,output_dtypes,inputs,
    cuda_header=r'''
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__device__ float bilinear_interp(
    const float *data,
    const float x,
    const float y,
    const int width,
    const int height)
{
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  float dist_x = static_cast<float>(x - x1);
  float dist_y = static_cast<float>(y - y1);
  float value11 = data[y1 * width + x1];
  float value12 = data[y2 * width + x1];
  float value21 = data[y1 * width + x2];
  float value22 = data[y2 * width + x2];
  float value = (1 - dist_x) * (1 - dist_y) * value11 +
            (1 - dist_x) * dist_y * value12 +
            dist_x * (1 - dist_y) * value21 +
            dist_x * dist_y * value22;
  return value;
}
__global__ void DeformablePSROIPoolForwardKernel(
    const int count,
    const float *bottom_data,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const float *bottom_rois, const float *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int output_dim,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class,
    float *top_data,
    float *top_count)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    const float *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    // Force too small ROIs to be 1x1
    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    float roi_height = max(roi_end_h - roi_start_h, 0.1);
    // Compute w and h at bottom
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);
    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);
    int part_h = floor(static_cast<float>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<float>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    float trans_x = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;
    float trans_y = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;
    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    float sum = 0;
    int count = 0;
    int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<float>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    const float *offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        float w = wstart + iw * sub_bin_size_w;
        float h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        float val = bilinear_interp(offset_bottom_data + c * height * width, w, h, width, height);
        sum += val;
        count++;
      }
    }
    top_data[index] = count == 0 ? static_cast<float>(0) : sum / count;
    top_count[index] = count;
  }
}
    ''',
    cuda_src=f'''
    const int no_trans = {no_trans};
    const float spatial_scale = {spatial_scale};
    const int output_dim = {output_dim};
    const int group_size = {group_size};
    const int pooled_size = {pooled_size};
    const int part_size = {part_size};
    const int sample_per_part = {sample_per_part};
    const float trans_std = {trans_std};
    '''+r'''
    @alias(input,in0)
    @alias(bbox,in1)
    @alias(trans,in2)
    @alias(top_count,out1)
  const int batch = input_shape0;
  const int channels = input_shape1;
  const int height = input_shape2;
  const int width = input_shape3;
  const int channels_trans = no_trans ? 2 : trans_shape1;
  const int num_bbox = bbox_shape0;
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
  long tmp = out_size % 512L==0? out_size/512L :out_size/512L+1L;
  dim3 grid(std::min(tmp, 4096L));
  dim3 block(512);
  DeformablePSROIPoolForwardKernel<<<grid, block>>>(
        out_size,
        input_p,
        spatial_scale,
        channels,
        height, width,
        pooled_height,
        pooled_width,
        bbox_p,
        trans_p,
        no_trans,
        trans_std,
        sample_per_part,
        output_dim,
        group_size,
        part_size,
        num_classes,
        channels_each_class,
        out_p,
        top_count_p);
    ''')
    return out,top_count



def dcn_v2_pooling_backward(grad_output,input,bbox,trans,output_count,no_trans,spatial_scale,output_dim,group_size,pooled_size,part_size,sample_per_part,trans_std):
    output_shape = [input.shape,trans.shape]
    output_dtype = [grad_output.dtype,trans.dtype]
    inputs = [grad_output,input,bbox,trans,output_count]
    input_grad,trans_grad = jt.code(output_shape,output_dtype,inputs,
    cuda_header=r'''
#include<cstdio>
#include<cstring>
#include<algorithm>
using namespace std;
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count,
    const float *top_diff,
    const float *top_count,
    const int num_rois,
    const float spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim,
    float *bottom_data_diff, float *bottom_trans_diff,
    const float *bottom_data,
    const float *bottom_rois,
    const float *bottom_trans,
    const int no_trans,
    const float trans_std,
    const int sample_per_part,
    const int group_size,
    const int part_size,
    const int num_classes,
    const int channels_each_class)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    const float *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    float roi_start_w = static_cast<float>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    float roi_start_h = static_cast<float>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    float roi_end_w = static_cast<float>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    float roi_end_h = static_cast<float>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    // Force too small ROIs to be 1x1
    float roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
    float roi_height = max(roi_end_h - roi_start_h, 0.1);
    // Compute w and h at bottom
    float bin_size_h = roi_height / static_cast<float>(pooled_height);
    float bin_size_w = roi_width / static_cast<float>(pooled_width);
    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);
    int part_h = floor(static_cast<float>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<float>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    float trans_x = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w] * trans_std;
    float trans_y = no_trans ? static_cast<float>(0) : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w] * trans_std;
    float wstart = static_cast<float>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    float hstart = static_cast<float>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    if (top_count[index] <= 0)
    {
      continue;
    }
    float diff_val = top_diff[index] / top_count[index];
    const float *offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
    float *offset_bottom_data_diff = bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<float>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    for (int ih = 0; ih < sample_per_part; ih++)
    {
      for (int iw = 0; iw < sample_per_part; iw++)
      {
        float w = wstart + iw * sub_bin_size_w;
        float h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5)
        {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        // backward on feature
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);
        float dist_x = w - x0, dist_y = h - y0;
        float q00 = (1 - dist_x) * (1 - dist_y);
        float q01 = (1 - dist_x) * dist_y;
        float q10 = dist_x * (1 - dist_y);
        float q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x0, q00 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x0, q01 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y0 * width + x1, q10 * diff_val);
        atomicAdd(offset_bottom_data_diff + bottom_index_base + y1 * width + x1, q11 * diff_val);
        if (no_trans)
        {
          continue;
        }
        float U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        float U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        float U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        float U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
        float diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y - U00 * (1 - dist_y)) * trans_std * diff_val;
        diff_x *= roi_width;
        float diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x - U00 * (1 - dist_x)) * trans_std * diff_val;
        diff_y *= roi_height;
        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h) * part_size + part_w, diff_x);
        atomicAdd(bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) * part_size + part_w, diff_y);
      }
    }
  }
}
    ''',
    cuda_src= f'''
    const int no_trans = {no_trans};
    const float spatial_scale = {spatial_scale};
    const int output_dim = {output_dim};
    const int group_size = {group_size};
    const int pooled_size = {pooled_size};
    const int part_size = {part_size};
    const int sample_per_part = {sample_per_part};
    const float trans_std = {trans_std};
    '''+r'''
    @alias(out_grad,in0)
    @alias(input,in1)
    @alias(bbox,in2)
    @alias(trans,in3)
    @alias(top_count,in4)
    @alias(input_grad,out0)
    @alias(trans_grad,out1)
    const int batch = input_shape0;
  const int channels = input_shape1;
  const int height = input_shape2;
  const int width = input_shape3;
  const int channels_trans = no_trans ? 2 : trans_shape1;
  const int num_bbox = bbox_shape0;
  auto pooled_height = pooled_size;
  auto pooled_width = pooled_size;
  long out_size = num_bbox * output_dim * pooled_height * pooled_width;
  const int num_classes = no_trans ? 1 : channels_trans / 2;
  const int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
  long tmp = out_size % 512L==0? out_size/512L :out_size/512L+1L;
  dim3 grid(std::min(tmp, 4096L));
  dim3 block(512);
  DeformablePSROIPoolBackwardAccKernel<<<grid, block>>>(
        out_size,
        out_grad_p,
        top_count_p,
        num_bbox,
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        output_dim,
        input_grad_p,
        trans_grad_p,
        input_p,
        bbox_p,
        trans_p,
        no_trans,
        trans_std,
        sample_per_part,
        group_size,
        part_size,
        num_classes,
        channels_each_class);
    ''')
    return input_grad,trans_grad


class DCN_V2_POOLING(jt.Function):
    def execute(self,input, rois, offset, spatial_scale,pooled_size,output_dim,no_trans,group_size=1,part_size=None,sample_per_part=4,trans_std=.0):
        self.spatial_scale = spatial_scale
        self.no_trans = int(no_trans)
        self.output_dim = output_dim
        self.group_size = group_size
        self.pooled_size = pooled_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std
        output,output_count = dcn_v2_pooling_forward(input,rois,offset,self.spatial_scale,self.pooled_size,self.output_dim,self.no_trans,self.group_size,self.part_size,self.sample_per_part,self.trans_std)
        self.output_count = output_count
        self.input = input
        self.rois = rois
        self.offset = offset
        return output


    def grad(self,grad_output):
        grad_input, grad_offset = dcn_v2_pooling_backward(grad_output,
                                                          self.input,
                                                          self.rois,
                                                          self.offset,
                                                          self.output_count,
                                                          self.no_trans,
                                                          self.spatial_scale,
                                                          self.output_dim,
                                                          self.group_size,
                                                          self.pooled_size,
                                                          self.part_size,
                                                          self.sample_per_part,
                                                          self.trans_std)
        return grad_input, None, grad_offset, None, None, None, None, None, None, None, None

dcn_v2_pooling = DCN_V2_POOLING.apply


class DeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = jt.zeros((out_channels, in_channels,*self.kernel_size))
        if bias:
            self.bias = jt.zeros((out_channels,))
        else:
            self.bias = np.zeros((out_channels,))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        nn.init.uniform_(self.weight,-stdv, stdv)

    def execute(self, x, offset):
        assert x.size(2) > self.kernel_size[0] and x.size(3) > self.kernel_size[1]
        mask_shape = list(offset.size())
        mask_shape[1] //= 2
        mask = jt.ones(mask_shape,x.dtype)
        return dcn_v2_conv(x, offset, mask,
                           self.weight,
                           jt.array(self.bias),
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = jt.zeros((out_channels, in_channels, *self.kernel_size))
        self.bias = jt.zeros((out_channels,))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        nn.init.constant_(self.bias,0.0)
        nn.init.uniform_(self.weight,-stdv,stdv)

    def execute(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] ==  offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == mask.shape[1]
        
        return dcn_v2_conv(input, offset, mask,
                           self.weight,
                           self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)

@HEADS.register_module()
class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv(self.in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset_mask.weight,0.0)
        nn.init.constant_(self.conv_offset_mask.bias,0.0)

    def execute(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = jt.chunk(out, 3, dim=1)
        offset = jt.contrib.concat((o1, o2), dim=1)
        mask = jt.sigmoid(mask)
        return dcn_v2_conv(input, offset, mask,
                           self.weight, self.bias,
                           self.stride,
                           self.padding,
                           self.dilation,
                           self.deformable_groups)



class DCNv2Pooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DCNv2Pooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.pooled_size = pooled_size
        self.output_dim = output_dim
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = pooled_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def execute(self, input, rois, offset):
        assert input.shape[1] == self.output_dim
        if self.no_trans:
            o_shape = (0,)+input.shape[1:]
            offset = jt.empty(o_shape) #jt.zeros_like(input)
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)


class DCNPooling(DCNv2Pooling):

    def __init__(self,
                 spatial_scale,
                 pooled_size,
                 output_dim,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_dim=1024):
        super(DCNPooling, self).__init__(spatial_scale,
                                         pooled_size,
                                         output_dim,
                                         no_trans,
                                         group_size,
                                         part_size,
                                         sample_per_part,
                                         trans_std)

        self.deform_fc_dim = deform_fc_dim

        if not no_trans:
            self.offset_mask_fc = nn.Sequential(
                nn.Linear(self.pooled_size * self.pooled_size *
                          self.output_dim, self.deform_fc_dim),
                nn.ReLU(),
                nn.Linear(self.deform_fc_dim, self.deform_fc_dim),
                nn.ReLU(),
                nn.Linear(self.deform_fc_dim, self.pooled_size *
                          self.pooled_size * 3)
            )
            nn.init.constant_(self.offset_mask_fc[4].weight,0.0)
            nn.init.constant_(self.offset_mask_fc[4].bias,0.0)

    def execute(self, input, rois):
        o_shape = (0,)+input.shape[1:]
        offset = jt.empty(o_shape) #jt.zeros_like(input)

        if not self.no_trans:

            # do roi_align first
            n = rois.shape[0]
            roi = dcn_v2_pooling(input, rois, offset,
                                 self.spatial_scale,
                                 self.pooled_size,
                                 self.output_dim,
                                 True,  # no trans
                                 self.group_size,
                                 self.part_size,
                                 self.sample_per_part,
                                 self.trans_std)

            # build mask and offset
            offset_mask = self.offset_mask_fc(roi.view(n, -1))
            offset_mask = offset_mask.view(
                n, 3, self.pooled_size, self.pooled_size)
            o1, o2, mask = jt.chunk(offset_mask, 3, dim=1)
            offset = jt.contrib.concat((o1, o2), dim=1)
            mask = jt.sigmoid(mask)

            # do pooling with offset and mask
            return dcn_v2_pooling(input, rois, offset,
                                  self.spatial_scale,
                                  self.pooled_size,
                                  self.output_dim,
                                  self.no_trans,
                                  self.group_size,
                                  self.part_size,
                                  self.sample_per_part,
                                  self.trans_std) * mask
        # only roi_align
        return dcn_v2_pooling(input, rois, offset,
                              self.spatial_scale,
                              self.pooled_size,
                              self.output_dim,
                              self.no_trans,
                              self.group_size,
                              self.part_size,
                              self.sample_per_part,
                              self.trans_std)


def test_conv():
    import numpy as np
    jt.flags.use_cuda=1
    input = jt.array(np.random.randn(2, 64, 128, 128).astype(np.float32))
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1,
              padding=1, deformable_groups=2)
    # print(dcn.weight.shape, input.shape)
    output = dcn(input)
    print(output)
    print(jt.grad(output.sum(),input))

def test_pool():
    import numpy as np
    jt.flags.use_cuda=1
    input = jt.array(np.random.randn(2, 32, 64, 64).astype(np.float32))
    batch_inds = jt.array(np.random.randint(2, size=(20, 1)).astype(np.int32))
    x = jt.array(np.random.randint(256, size=(20, 1))).float32()
    y = jt.array(np.random.randint(256, size=(20, 1))).float32()
    w =jt.array(np.random.randint(64, size=(20, 1))).float32()
    h = jt.array(np.random.randint(64, size=(20, 1))).float32()
    rois = jt.contrib.concat((batch_inds, x, y, x + w, y + h), dim=1)
    offset = jt.array(np.random.randn(20, 2, 7, 7).astype(np.float32))

    # normal roi_align
    pooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                           pooled_size=7,
                           output_dim=32,
                           no_trans=True,
                           group_size=1,
                           trans_std=0.1)
    
    # deformable pooling
    dpooling = DCNv2Pooling(spatial_scale=1.0 / 4,
                            pooled_size=7,
                            output_dim=32,
                            no_trans=False,
                            group_size=1,
                            trans_std=0.1)

    out = pooling(input, rois, offset)
    dout = dpooling(input, rois, offset)
    print(out)
    print(dout)
    print(jt.grad(out.sum(),input))
    print(jt.grad(dout.sum(),input))



if __name__ == '__main__':
    test_conv()
    test_pool()