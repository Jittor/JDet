import jittor as jt
import os

from jittor.nn import _pair

# cuda codes
# warning: nthreads > MAX_INT?

_kernel_loop_head = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
#define THREADS_PER_BLOCK 1024
inline int GET_BLOCKS(const int N) {
    return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}
'''

_subtraction_zeropad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction_zeropad_forward_kernel(
    const int nthreads, const T* bottom_data, T* top_data, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % input_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int h_in_center = -pad_h + h * stride_h + (kernel_h - 1) / 2 * dilation_h;
    const int w_in_center = -pad_w + w * stride_w + (kernel_w - 1) / 2 * dilation_w;
    const int offset_center = ((n * input_channels + c) * bottom_height + h_in_center) * bottom_width + w_in_center;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
          top_data[offset_top] = bottom_data[offset_center] - bottom_data[offset_bottom];
        }
        else
          top_data[offset_top] = bottom_data[offset_center];
      }
    }
  }
}
'''

_subtraction_zeropad_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction_zeropad_input_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % input_channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += -top_diff[offset_top];
          }
        }
      }
    }
    if (((h % stride_h) == 0) && ((w % stride_w) == 0)) {
      const int h_out = h / stride_h;
      const int w_out = w / stride_w;
      for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
          const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
          value += top_diff[offset_top];
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_subtraction_refpad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction_refpad_forward_kernel(
    const int nthreads, const T* bottom_data, T* top_data, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
  ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % input_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int h_in_center = -pad_h + h * stride_h + (kernel_h - 1) / 2 * dilation_h;
    const int w_in_center = -pad_w + w * stride_w + (kernel_w - 1) / 2 * dilation_w;
    const int offset_center = ((n * input_channels + c) * bottom_height + h_in_center) * bottom_width + w_in_center;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        int h_in = -pad_h + h * stride_h + kh * dilation_h;
        int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        int offset_bottom;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
        }
        else {
          if (h_in < 0) h_in = -h_in;
          if (h_in >= bottom_height) h_in = 2 * (bottom_height - 1) - h_in;
          if (w_in < 0) w_in = -w_in;
          if (w_in >= bottom_width) w_in = 2 * (bottom_width - 1) - w_in;
          offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
        }
        top_data[offset_top] = bottom_data[offset_center] - bottom_data[offset_bottom];
      }
    }
  }
}
'''

_subtraction_refpad_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction_refpad_input_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / (bottom_height + 2 * pad_h) / (bottom_width + 2 * pad_w);
    const int c = (index / (bottom_height + 2 * pad_h) / (bottom_width + 2 * pad_w)) % input_channels;
    const int h = (index / (bottom_width + 2 * pad_w)) % (bottom_height + 2 * pad_h);
    const int w = index % (bottom_width + 2 * pad_w);
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h - kh * dilation_h;
        const int w_out_s = w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += -top_diff[offset_top];
          }
        }
      }
    }
    const int hh = h - pad_h;
    const int ww = w - pad_w;
    if ((hh >= 0) && (hh < bottom_height) && (ww >= 0) && (ww < bottom_width)) {
      if (((hh % stride_h) == 0) && ((ww % stride_w) == 0)) {
        const int h_out = hh / stride_h;
        const int w_out = ww / stride_w;
        for (int kh = 0; kh < kernel_h; ++kh) {
          for (int kw = 0; kw < kernel_w; ++kw) {
            const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_subtraction2_zeropad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_zeropad_forward_kernel(
    int nthreads, const T* bottom1_data, const T* bottom2_data, T* top_data, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
  ) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % input_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int h_in_center = -pad_h + h * stride_h + (kernel_h - 1) / 2 * dilation_h;
    const int w_in_center = -pad_w + w * stride_w + (kernel_w - 1) / 2 * dilation_w;
    const int offset_center = ((n * input_channels + c) * bottom_height + h_in_center) * bottom_width + w_in_center;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
          top_data[offset_top] = bottom1_data[offset_center] - bottom2_data[offset_bottom];
        }
        else
          top_data[offset_top] = bottom1_data[offset_center];
      }
    }
  }
}
'''

_subtraction2_zeropad_input1_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_zeropad_input1_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % input_channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    T value = 0;
    if (((h % stride_h) == 0) && ((w % stride_w) == 0)) {
      const int h_out = h / stride_h;
      const int w_out = w / stride_w;
      for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
          const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
          value += top_diff[offset_top];
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_subtraction2_zeropad_input2_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_zeropad_input2_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % input_channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += -top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_subtraction2_refpad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_refpad_forward_kernel(
    int nthreads, const T* bottom1_data, const T* bottom2_data, T* top_data, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
  ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % input_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int h_in_center = -pad_h + h * stride_h + (kernel_h - 1) / 2 * dilation_h;
    const int w_in_center = -pad_w + w * stride_w + (kernel_w - 1) / 2 * dilation_w;
    const int offset_center = ((n * input_channels + c) * bottom_height + h_in_center) * bottom_width + w_in_center;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        int h_in = -pad_h + h * stride_h + kh * dilation_h;
        int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        int offset_bottom;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
        }
        else {
          if (h_in < 0) h_in = -h_in;
          if (h_in >= bottom_height) h_in = 2 * (bottom_height - 1) - h_in;
          if (w_in < 0) w_in = -w_in;
          if (w_in >= bottom_width) w_in = 2 * (bottom_width - 1) - w_in;
          offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
        }
        top_data[offset_top] = bottom1_data[offset_center] - bottom2_data[offset_bottom];
      }
    }
  }
}
'''

_subtraction2_refpad_input1_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_refpad_input1_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % input_channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    T value = 0;
    if (((h % stride_h) == 0) && ((w % stride_w) == 0)) {
      const int h_out = h / stride_h;
      const int w_out = w / stride_w;
      for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
          const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
          value += top_diff[offset_top];
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_subtraction2_refpad_input2_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void subtraction2_refpad_input2_backward_kernel(
    const int nthreads, const T* const top_diff, T* bottom_diff, const int input_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / input_channels / (bottom_height + 2 * pad_h) / (bottom_width + 2 * pad_w);
    const int c = (index / (bottom_height + 2 * pad_h) / (bottom_width + 2 * pad_w)) % input_channels;
    const int h = (index / (bottom_width + 2 * pad_w)) % (bottom_height + 2 * pad_h);
    const int w = index % (bottom_width + 2 * pad_w);
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h - kh * dilation_h;
        const int w_out_s = w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int offset_top = ((n * input_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += -top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''


# classes
# TODO: inplace codes like f'''...''' into r'''...'''
# TODO: check backward.
def _tuple_numel(shape):
  return shape[0] * shape[1] * shape[2] * shape[3]

class SubtractionZeropad(jt.Function):
    def execute(self, input, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input = input
        assert len(input.shape) == 4 and jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output_shape = (batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
        nthreads = batch_size * input_channels * output_height * output_width
        subtraction_zeropad_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction_zeropad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input.dtype, [input], cuda_header=_subtraction_zeropad_forward_header, cuda_src=subtraction_zeropad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input = self.input
        assert jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        nthreads = input.numel()
        subtraction_zeropad_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction_zeropad_input_backward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input = jt.code(input.size(), grad_output.dtype, [grad_output], cuda_header=_subtraction_zeropad_backward_header, cuda_src=subtraction_zeropad_backward_src)
        return grad_input, None, None, None, None

class SubtractionRefpad(jt.Function):
    def execute(self, input, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input = input
        assert len(input.shape) == 4 and jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output_shape = (batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
        nthreads = batch_size * input_channels * output_height * output_width
        subtraction_refpad_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction_refpad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input.dtype, [input], cuda_header=_subtraction_refpad_forward_header, cuda_src=subtraction_refpad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input = self.input
        assert jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        grad_shape = (batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
        nthreads = _tuple_numel(grad_shape)
        subtraction_refpad_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction_refpad_input_backward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input = jt.code(grad_shape, grad_output.dtype, [grad_output], cuda_header=_subtraction_refpad_backward_header, cuda_src=subtraction_refpad_backward_src)
        grad_input[:, :, padding[0] + 1:2 * padding[0] + 1, :] += jt.flip(grad_input[:, :, :padding[0], :], dim=2)
        grad_input[:, :, input_height - 1:input_height + padding[0] - 1, :] += jt.flip(grad_input[:, :, input_height + padding[0]:, :], dim=2)
        grad_input[:, :, :, padding[1] + 1:2 * padding[1] + 1] += jt.flip(grad_input[:, :, :, :padding[1]], dim=3)
        grad_input[:, :, :, input_width - 1:input_width + padding[1] - 1] += jt.flip(grad_input[:, :, :, input_width + padding[1]:], dim=3)
        grad_input = grad_input[:, :, padding[0]:padding[0] + input_height, padding[1]:padding[1] + input_width]
        return grad_input, None, None, None, None



class Subtraction2Zeropad(jt.Function):
    def execute(self, input1, input2, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input1, self.input2 = input1, input2
        assert len(input1.shape) == 4 and jt.flags.use_cuda
        assert input1.size() == input2.size()
        batch_size, input_channels, input_height, input_width = input1.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output_shape = (batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
        nthreads = batch_size * input_channels * output_height * output_width
        subtraction2_zeropad_src = f'''
            @alias(input1,in0);
            @alias(input2,in1);
            @alias(output,out0);
            subtraction2_zeropad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input1_p, input2_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input1.dtype, [input1, input2], cuda_header=_subtraction2_zeropad_forward_header, cuda_src=subtraction2_zeropad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input1, input2 = self.input1, self.input2
        assert jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input1.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        nthreads1, nthreads2 = input1.numel(), input2.numel()
        subtraction2_zeropad_input1_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction2_zeropad_input1_backward_kernel<<<GET_BLOCKS({nthreads1}), THREADS_PER_BLOCK>>>(
                {nthreads1}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        subtraction2_zeropad_input2_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction2_zeropad_input2_backward_kernel<<<GET_BLOCKS({nthreads2}), THREADS_PER_BLOCK>>>(
                {nthreads2}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input1 = jt.code(input1.size(), grad_output.dtype, [grad_output], cuda_header=_subtraction2_zeropad_input1_backward_header, cuda_src=subtraction2_zeropad_input1_backward_src)
        grad_input2 = jt.code(input2.size(), grad_output.dtype, [grad_output], cuda_header=_subtraction2_zeropad_input2_backward_header, cuda_src=subtraction2_zeropad_input2_backward_src)
        return grad_input1, grad_input2, None, None, None, None

class Subtraction2Refpad(jt.Function):
    def execute(self, input1, input2, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input1, self.input2 = input1, input2
        assert len(input1.shape) == 4 and jt.flags.use_cuda
        assert input1.size() == input2.size()
        batch_size, input_channels, input_height, input_width = input1.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        output_shape = (batch_size, input_channels, kernel_size[0] * kernel_size[1], output_height * output_width)
        nthreads = batch_size * input_channels * output_height * output_width
        subtraction2_refpad_src = f'''
            @alias(input1,in0);
            @alias(input2,in1);
            @alias(output,out0);
            subtraction2_refpad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input1_p, input2_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input1.dtype, [input1, input2], cuda_header=_subtraction2_refpad_forward_header, cuda_src=subtraction2_refpad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input1, input2 = self.input1, self.input2
        assert jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input1.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        grad_shape2 = (batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
        nthreads2 = _tuple_numel(grad_shape2)
        nthreads1 = input1.numel()
        subtraction2_refpad_input1_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction2_refpad_input1_backward_kernel<<<GET_BLOCKS({nthreads1}), THREADS_PER_BLOCK>>>(
                {nthreads1}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        subtraction2_refpad_input2_backward_src = f'''
            @alias(input,in0);
            @alias(output,out0);
            subtraction2_refpad_input2_backward_kernel<<<GET_BLOCKS({nthreads2}), THREADS_PER_BLOCK>>>(
                {nthreads2}, input_p, output_p, {input_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input1 = jt.code(input1.size(), grad_output.dtype, [grad_output], cuda_header=_subtraction2_refpad_input1_backward_header, cuda_src=subtraction2_refpad_input1_backward_src)
        grad_input2 = jt.code(grad_shape2, grad_output.dtype, [grad_output], cuda_header=_subtraction2_refpad_input2_backward_header, cuda_src=subtraction2_refpad_input2_backward_src)
        grad_input2[:, :, padding[0] + 1:2 * padding[0] + 1, :] += jt.flip(grad_input2[:, :, :padding[0], :], dim=2)
        grad_input2[:, :, input_height - 1:input_height + padding[0] - 1, :] += jt.flip(grad_input2[:, :, input_height + padding[0]:, :], dim=2)
        grad_input2[:, :, :, padding[1] + 1:2 * padding[1] + 1] += jt.flip(grad_input2[:, :, :, :padding[1]], dim=3)
        grad_input2[:, :, :, input_width - 1:input_width + padding[1] - 1] += jt.flip(grad_input2[:, :, :, input_width + padding[1]:], dim=3)
        grad_input2 = grad_input2[:, :, padding[0]:padding[0] + input_height, padding[1]:padding[1] + input_width]
        return grad_input1, grad_input2, None, None, None, None


# functions
def subtraction_zeropad(input, kernel_size=3, stride=1, padding=0, dilation=1):
    assert len(input.size()) == 4
    if jt.flags.use_cuda == 1:
        out = SubtractionZeropad.apply(input, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def subtraction_refpad(input, kernel_size=3, stride=1, padding=0, dilation=1):
    assert len(input.size()) == 4
    if jt.flags.use_cuda == 1:
        out = SubtractionRefpad.apply(input, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def subtraction2_zeropad(input1, input2, kernel_size=3, stride=1, padding=0, dilation=1):
    assert len(input1.size()) == 4
    if jt.flags.use_cuda == 1:
        out = Subtraction2Zeropad.apply(input1, input2, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def subtraction2_refpad(input1, input2, kernel_size=3, stride=1, padding=0, dilation=1):
    assert len(input1.size()) == 4
    if jt.flags.use_cuda == 1:
        out = Subtraction2Refpad.apply(input1, input2, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def subtraction(input, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert len(input.size()) == 4 and pad_mode in [0, 1]
    if jt.flags.use_cuda == 1:
        if pad_mode == 0:
            out = subtraction_zeropad(input, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = subtraction_refpad(input, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out


def subtraction2(input1, input2, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert len(input1.size()) == 4 and len(input2.size()) == 4 and pad_mode in [0, 1]
    if jt.flags.use_cuda == 1:
        if pad_mode == 0:
            out = subtraction2_zeropad(input1, input2, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = subtraction2_refpad(input1, input2, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

# unit tests
def test_subtraction_zeropad():
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c, in_height, in_width = 2, 8, 5, 5
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)

    y1 = subtraction_zeropad(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    unfold_i = jt.nn.unfold(x, kernel_size=1, dilation=dilation, padding=0, stride=stride)
    unfold_j = jt.nn.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    y2 = unfold_i.reshape((n, c, 1, out_height * out_width)) - unfold_j.reshape((n, c, pow(kernel_size, 2), out_height * out_width))
    assert (y1 - y2).abs().max() < 1e-9

    gx1 = jt.grad(y1.mean(), x)[0]
    gx2 = jt.grad(y2.mean(), x)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    print('subtraction_zeropad passed')

def test_subtraction_refpad():
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c, in_height, in_width = 2, 8, 5, 5
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)

    y1 = subtraction_refpad(x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    pad = jt.nn.ReflectionPad2d(padding)
    unfold_i = jt.nn.unfold(x, kernel_size=1, dilation=dilation, padding=0, stride=stride)
    unfold_j = jt.nn.unfold(pad(x), kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
    y2 = unfold_i.reshape((n, c, 1, out_height * out_width)) - unfold_j.reshape((n, c, pow(kernel_size, 2), out_height * out_width))
    assert (y1 - y2).abs().max() < 1e-9

    gx1 = jt.grad(y1.mean(), x)[0]
    gx2 = jt.grad(y2.mean(), x)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    print('subtraction_refpad passed')

def test_subtraction2_zeropad():
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c, in_height, in_width = 2, 8, 9, 9
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x1 = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)
    x2 = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)

    y1 = subtraction2_zeropad(x1, x2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    unfold_i = jt.nn.unfold(x1, kernel_size=1, dilation=dilation, padding=0, stride=stride)
    unfold_j = jt.nn.unfold(x2, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    y2 = unfold_i.reshape((n, c, 1, out_height * out_width)) - unfold_j.reshape((n, c, pow(kernel_size, 2), out_height * out_width))
    assert (y1 - y2).abs().max() < 1e-9

    gx11 = jt.grad(y1.mean(), x1)[0]
    gx12 = jt.grad(y1.mean(), x2)[0]
    gx21 = jt.grad(y2.mean(), x1)[0]
    gx22 = jt.grad(y2.mean(), x2)[0]
    assert (gx11 - gx21).abs().max() < 1e-9
    assert (gx12 - gx22).abs().max() < 1e-9

    print('subtraction2_zeropad passed')

def test_subtraction2_refpad():
    kernel_size, stride, dilation = 5, 4, 2  # 3, 1, 1
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c, in_height, in_width = 2, 8, 9, 9
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x1 = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)
    x2 = jt.randn(n, c, in_height, in_width, requires_grad=True, dtype=jt.float64)

    y1 = subtraction2_refpad(x1, x2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    pad = jt.nn.ReflectionPad2d(padding)
    unfold_i = jt.nn.unfold(x1, kernel_size=1, dilation=dilation, padding=0, stride=stride)
    unfold_j = jt.nn.unfold(pad(x2), kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
    y2 = unfold_i.reshape((n, c, 1, out_height * out_width)) - unfold_j.reshape((n, c, pow(kernel_size, 2), out_height * out_width))
    assert (y1 - y2).abs().max() < 1e-9

    gx11 = jt.grad(y1.mean(), x1)[0]
    gx12 = jt.grad(y1.mean(), x2)[0]
    gx21 = jt.grad(y2.mean(), x1)[0]
    gx22 = jt.grad(y2.mean(), x2)[0]
    assert (gx11 - gx21).abs().max() < 1e-9
    assert (gx12 - gx22).abs().max() < 1e-9

    print('subtraction2_refpad passed')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    jt.flags.use_cuda = 1
    print("start...")
    test_subtraction_zeropad()
    test_subtraction_refpad()
    test_subtraction2_zeropad()
    test_subtraction2_refpad()
    print("done.")