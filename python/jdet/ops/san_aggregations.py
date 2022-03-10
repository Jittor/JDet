import jittor as jt
import os
from jittor.nn import _pair


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

_aggregation_zeropad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_zeropad_forward_kernel(
    const int nthreads, const T* bottom_data, const T* weight_data, T* top_data,
    const int input_channels, const int weight_channels,
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
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset_bottom = ((n * input_channels + c) * bottom_height + h_in) * bottom_width + w_in;
          const int offset_weight = ((n * weight_channels + c % weight_channels) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
          value += weight_data[offset_weight] * bottom_data[offset_bottom];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_zeropad_input_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_zeropad_input_backward_kernel(
    const int nthreads, const T* const top_diff, const T* const weight_data, T* bottom_diff,
    const int input_channels, const int weight_channels,
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
            const int offset_top = ((n * input_channels + c) * top_height + h_out) * top_width + w_out;
            const int offset_weight = ((n * weight_channels + c % weight_channels) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += weight_data[offset_weight] * top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_aggregation_zeropad_weight_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_zeropad_weight_backward_kernel(
    const int nthreads, const T* const top_diff, const T* const bottom_data, T* weight_diff,
    const int input_channels, const int weight_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / weight_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % weight_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_weight = ((n * weight_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        T value = 0;
        if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
          for (int cc = c; cc < input_channels; cc += weight_channels) {
            const int offset_bottom = ((n * input_channels + cc) * bottom_height + h_in) * bottom_width + w_in;
            const int offset_top = ((n * input_channels + cc) * top_height + h) * top_width + w;
            value += bottom_data[offset_bottom] * top_diff[offset_top];
          }
        }
        weight_diff[offset_weight] = value;
      }
    }
  }
}
'''

_aggregation_refpad_forward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_refpad_forward_kernel(
    const int nthreads, const T* bottom_data, const T* weight_data, T* top_data,
    const int input_channels, const int weight_channels,
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
    T value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        int h_in = -pad_h + h * stride_h + kh * dilation_h;
        int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_weight = ((n * weight_channels + c % weight_channels) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
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
        value += weight_data[offset_weight] * bottom_data[offset_bottom];
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_refpad_input_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_refpad_input_backward_kernel(
    const int nthreads, const T* const top_diff, const T* const weight_data, T* bottom_diff,
    const int input_channels, const int weight_channels,
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
        if ((h_out_s % stride_h == 0) && (w_out_s % stride_w == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height) && (w_out >= 0) && (w_out < top_width)) {
            const int offset_top = ((n * input_channels + c) * top_height + h_out) * top_width + w_out;
            const int offset_weight = ((n * weight_channels + c % weight_channels) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h_out * top_width + w_out;
            value += weight_data[offset_weight] * top_diff[offset_top];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_aggregation_refpad_weight_backward_header = _kernel_loop_head + r'''
template <typename T>
__global__ void aggregation_refpad_weight_backward_kernel(
    const int nthreads, const T* const top_diff, const T* const bottom_data, T* weight_diff,
    const int input_channels, const int weight_channels,
    const int bottom_height, const int bottom_width,
    const int top_height, const int top_width,
    const int pad_h, const int stride_h, const int kernel_h, const int dilation_h,
    const int pad_w, const int stride_w, const int kernel_w, const int dilation_w
    ) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / weight_channels / top_height / top_width;
    const int c = (index / top_height / top_width) % weight_channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        int h_in = -pad_h + h * stride_h + kh * dilation_h;
        int w_in = -pad_w + w * stride_w + kw * dilation_w;
        const int offset_weight = ((n * weight_channels + c) * kernel_h * kernel_w + (kh * kernel_w + kw)) * top_height * top_width + h * top_width + w;
        T value = 0;
        for (int cc = c; cc < input_channels; cc += weight_channels) {
          const int offset_top = ((n * input_channels + cc) * top_height + h) * top_width + w;
          int offset_bottom;
          if ((h_in >= 0) && (h_in < bottom_height) && (w_in >= 0) && (w_in < bottom_width)) {
            offset_bottom = ((n * input_channels + cc) * bottom_height + h_in) * bottom_width + w_in;
          }
          else {
            if (h_in < 0) h_in = -h_in;
            if (h_in >= bottom_height) h_in = 2 * (bottom_height - 1) - h_in;
            if (w_in < 0) w_in = -w_in;
            if (w_in >= bottom_width) w_in = 2 * (bottom_width - 1) - w_in;
            offset_bottom = ((n * input_channels + cc) * bottom_height + h_in) * bottom_width + w_in;
          }
          value += bottom_data[offset_bottom] * top_diff[offset_top];
        }
        weight_diff[offset_weight] = value;
      }
    }
  }
}
'''

def _tuple_numel(shape):
  return shape[0] * shape[1] * shape[2] * shape[3]

class AggregationZeropad(jt.Function):
    def execute(self, input, weight, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input_, self.weight_ = input, weight
        assert len(input.shape) == 4 and jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_channels, weight_height, weight_width = weight.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        assert output_height * output_width == weight_width
        output_shape = (batch_size, input_channels, output_height, output_width)
        nthreads = _tuple_numel(output_shape)
        aggregation_zeropad_src = f'''
            @alias(input,in0);
            @alias(weight,in1);
            @alias(output,out0);
            aggregation_zeropad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, weight_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input.dtype, [input, weight], cuda_header=_aggregation_zeropad_forward_header, cuda_src=aggregation_zeropad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input, weight = self.input_, self.weight_
        assert jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_channels, weight_height, weight_width = weight.size()
        output_height, output_width = grad_output.size()[2:]
        nthreads_input = input.numel()
        nthreads_weight = weight.numel() // weight.shape[2]
        aggregation_zeropad_input_backward_src = f'''
            @alias(diff,in0);
            @alias(weight,in1);
            @alias(output,out0);
            aggregation_zeropad_input_backward_kernel<<<GET_BLOCKS({nthreads_input}), THREADS_PER_BLOCK>>>(
                {nthreads_input}, diff_p, weight_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        aggregation_zeropad_weight_backward_src = f'''
            @alias(diff,in0);
            @alias(input,in1);
            @alias(output,out0);
            aggregation_zeropad_weight_backward_kernel<<<GET_BLOCKS({nthreads_weight}), THREADS_PER_BLOCK>>>(
                {nthreads_weight}, diff_p, input_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input = jt.code(input.size(), grad_output.dtype, [grad_output, weight], cuda_header=_aggregation_zeropad_input_backward_header, cuda_src=aggregation_zeropad_input_backward_src)
        grad_weight = jt.code(weight.size(), grad_output.dtype, [grad_output, input], cuda_header=_aggregation_zeropad_weight_backward_header, cuda_src=aggregation_zeropad_weight_backward_src)
        return grad_input, grad_weight, None, None, None, None

class AggregationRefpad(jt.Function):
    def execute(self, input, weight, kernel_size, stride, padding, dilation):
        kernel_size, stride, padding, dilation = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation)
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.input_, self.weight_ = input, weight
        assert len(input.shape) == 4 and jt.flags.use_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_channels, weight_height, weight_width = weight.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        assert output_height * output_width == weight_width
        output_shape = (batch_size, input_channels, output_height, output_width)
        nthreads = _tuple_numel(output_shape)
        aggregation_refpad_src = f'''
            @alias(input,in0);
            @alias(weight,in1);
            @alias(output,out0);
            aggregation_refpad_forward_kernel<<<GET_BLOCKS({nthreads}), THREADS_PER_BLOCK>>>(
                {nthreads}, input_p, weight_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        return jt.code(output_shape, input.dtype, [input, weight], cuda_header=_aggregation_refpad_forward_header, cuda_src=aggregation_refpad_src)

    def grad(self, grad_output):
        kernel_size, stride, padding, dilation = self.kernel_size, self.stride, self.padding, self.dilation
        input, weight = self.input_, self.weight_
        batch_size, input_channels, input_height, input_width = input.size()
        _, weight_channels, weight_height, weight_width = weight.size()
        output_height, output_width = grad_output.shape[2:]
        grad_input_shape = (batch_size, input_channels, input_height + 2 * padding[0], input_width + 2 * padding[1])
        nthreads_input = _tuple_numel(grad_input_shape)
        nthreads_weight = weight.numel() // weight.shape[2]
        aggregation_refpad_input_backward_src = f'''
            @alias(diff,in0);
            @alias(weight,in1);
            @alias(output,out0);
            aggregation_refpad_input_backward_kernel<<<GET_BLOCKS({nthreads_input}), THREADS_PER_BLOCK>>>(
                {nthreads_input}, diff_p, weight_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        aggregation_refpad_weight_backward_src = f'''
            @alias(diff,in0);
            @alias(input,in1);
            @alias(output,out0);
            aggregation_refpad_weight_backward_kernel<<<GET_BLOCKS({nthreads_weight}), THREADS_PER_BLOCK>>>(
                {nthreads_weight}, diff_p, input_p, output_p,
                {input_channels}, {weight_channels},
                {input_height}, {input_width},
                {output_height}, {output_width},
                {padding[0]}, {stride[0]}, {kernel_size[0]}, {dilation[0]},
                {padding[1]}, {stride[1]}, {kernel_size[1]}, {dilation[1]}
            );
        '''
        grad_input = jt.code(grad_input_shape, grad_output.dtype, [grad_output, weight], cuda_header=_aggregation_refpad_input_backward_header, cuda_src=aggregation_refpad_input_backward_src)
        grad_weight = jt.code(weight.size(), grad_output.dtype, [grad_output, input], cuda_header=_aggregation_refpad_weight_backward_header, cuda_src=aggregation_refpad_weight_backward_src)
        grad_input[:, :, padding[0] + 1:2 * padding[0] + 1, :] += jt.flip(grad_input[:, :, :padding[0], :], dim=2)
        grad_input[:, :, input_height - 1:input_height + padding[0] - 1, :] += jt.flip(grad_input[:, :, input_height + padding[0]:, :], dim=2)
        grad_input[:, :, :, padding[1] + 1:2 * padding[1] + 1] += jt.flip(grad_input[:, :, :, :padding[1]], dim=3)
        grad_input[:, :, :, input_width - 1:input_width + padding[1] - 1] += jt.flip(grad_input[:, :, :, input_width + padding[1]:], dim=3)
        grad_input = grad_input[:, :, padding[0]:padding[0]+input_height, padding[1]:padding[1]+input_width]
        return grad_input, grad_weight, None, None, None, None


def aggregation_zeropad(input, weight, kernel_size=3, stride=1, padding=0, dilation=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[1] == 0)
    if jt.flags.use_cuda == 1:
        out = AggregationZeropad.apply(input, weight, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def aggregation_refpad(input, weight, kernel_size=3, stride=1, padding=0, dilation=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[1] == 0)
    if jt.flags.use_cuda == 1:
        out = AggregationRefpad.apply(input, weight, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def aggregation(input, weight, kernel_size=3, stride=1, padding=0, dilation=1, pad_mode=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % weight.shape[1] == 0) and pad_mode in [0, 1]
    if jt.flags.use_cuda == 1:
        if pad_mode == 0:
            out = aggregation_zeropad(input, weight, kernel_size, stride, padding, dilation)
        elif pad_mode == 1:
            out = aggregation_refpad(input, weight, kernel_size, stride, padding, dilation)
    else:
        raise NotImplementedError
    return out

def test_aggregation_zeropad():
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, c_w, in_height, in_width = 2, 8, 4, 9, 9
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x = jt.randn(n, c_x, in_height, in_width, requires_grad=True, dtype=jt.float64)
    w = jt.randn(n, c_w, pow(kernel_size, 2), out_height * out_width, requires_grad=True, dtype=jt.float64)

    y1 = aggregation_zeropad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    unfold_j = jt.nn.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    x2 = unfold_j.reshape((n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width))
    y2 = (jt.unsqueeze(w, 1) * x2).sum(dim=-2).reshape((n, c_x, out_height, out_width))
 
    assert (y1 - y2).abs().max() < 1e-9
    gx1 = jt.grad(y1.mean(), x)[0]
    gx2 = jt.grad(y2.mean(), x)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    gw1 = jt.grad(y1.mean(), w)[0]
    gw2 = jt.grad(y2.mean(), w)[0]
    assert (gw1 - gw2).abs().max() < 1e-9

    print('aggregation_zeropad passed')

def test_aggregation_refpad():
    kernel_size, stride, dilation = 5, 4, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, c_w, in_height, in_width = 2, 8, 4, 5, 5
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    x = jt.randn(n, c_x, in_height, in_width, requires_grad=True, dtype=jt.float64)
    w = jt.randn(n, c_w, pow(kernel_size, 2), out_height * out_width, requires_grad=True, dtype=jt.float64)

    y1 = aggregation_refpad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    pad = jt.nn.ReflectionPad2d(padding)
    unfold_j = jt.nn.unfold(pad(x), kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
    x2 = unfold_j.reshape((n, c_x // c_w, c_w, pow(kernel_size, 2), out_height * out_width))
    y2 = (jt.unsqueeze(w, 1) * x2).sum(-2).reshape((n, c_x, out_height, out_width))
    assert (y1 - y2).abs().max() < 1e-9

    gx1 = jt.grad(y1.mean(), x)[0]
    gx2 = jt.grad(y2.mean(), x)[0]
    assert (gx1 - gx2).abs().max() < 1e-9

    gw1 = jt.grad(y1.mean(), w)[0]
    gw2 = jt.grad(y2.mean(), w)[0]
    assert (gw1 - gw2).abs().max() < 1e-9

    print('aggregation_refpad passed')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    jt.flags.use_cuda = 1
    print("start...")
    test_aggregation_zeropad()
    test_aggregation_refpad()
    print("done.")