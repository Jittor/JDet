# Modified from
# https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cu

import jittor as jt
from jittor import Function
import numpy as np

HEADER=r"""
#define MAX_SHARED_SCALAR_T 6144 
#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return std::min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void chamfer_distance_forward_cuda_kernel(int b, int n,
                                                     const scalar_t* xyz, int m,
                                                     const scalar_t* xyz2,
                                                     scalar_t* result,
                                                     int* result_i) {
  __shared__ scalar_t buf[MAX_SHARED_SCALAR_T];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += THREADS_PER_BLOCK) {
      int end_k = min(m, k2 + THREADS_PER_BLOCK) - k2;
      for (int j = threadIdx.x; j < end_k * 2; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 2 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 2 + 0];
        scalar_t y1 = xyz[(i * n + j) * 2 + 1];
        int best_i = 0;
        scalar_t best = 1e10;
        int end_ka = end_k & (~2);
        if (end_ka == THREADS_PER_BLOCK) {
          for (int k = 0; k < THREADS_PER_BLOCK; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              scalar_t x2 = buf[(k + j) * 2] - x1;
              scalar_t y2 = buf[(k + j) * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + j;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 2 + 0] - x1;
          scalar_t y2 = buf[k * 2 + 1] - y1;
          scalar_t d = x2 * x2 + y2 * y2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel(
    int b, int n, const scalar_t* xyz1, int m, const scalar_t* xyz2,
    const scalar_t* grad_dist1, const int* idx1, scalar_t* grad_xyz1,
    scalar_t* grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x; j < n; j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 2 + 1];
      scalar_t g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}
"""

def chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2):
    src = f"""
    const int batch_size = {xyz1.size(0)};
    const int n = {xyz1.size(1)};
    const int m = {xyz2.size(1)};
    chamfer_distance_forward_cuda_kernel<<<GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK >>>(
                batch_size, n, in0_p, m,
                in1_p, out0_p, out2_p);
    chamfer_distance_forward_cuda_kernel<<<GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK >>>(
                batch_size, m, in1_p, n,
                in0_p, out1_p, out3_p);
    """
    return jt.code(
      outputs=[dist1, dist2, idx1, idx2],
      inputs=[xyz1, xyz2],
      cuda_header=HEADER,
      cuda_src=src)

def chamfer_distance_backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2, grad_xyz1, grad_xyz2):
    src=f"""
    const int batch_size = {xyz1.size(0)};
    const int n = {xyz1.size(1)};
    const int m = {xyz2.size(1)};
    chamfer_distance_backward_cuda_kernel<<<GET_BLOCKS(batch_size * n), THREADS_PER_BLOCK / 2 >>>(
                batch_size, m, in0_p, n,
                in1_p, in4_p,
                in2_p, out0_p,
                out1_p);
                
    chamfer_distance_backward_cuda_kernel<<<GET_BLOCKS(batch_size * m), THREADS_PER_BLOCK / 2 >>>(
                batch_size, n, in1_p, m,
                in0_p, in5_p,
                in3_p, out1_p,
                out0_p);
    """
    return jt.code(
      outputs=[grad_xyz1, grad_xyz2],
      inputs=[xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2],
      cuda_header=HEADER,
      cuda_src=src)


class ChamferDistanceFunction(Function):
    """This is an implementation of the 2D Chamfer Distance.
    It has been used in the paper `Oriented RepPoints for Aerial Object
    Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
    """

    def execute(self, xyz1, xyz2):
        """
        Args:
            xyz1 (Tensor): Point set with shape (B, N, 2).
            xyz2 (Tensor): Point set with shape (B, N, 2).
        Returns:
            Sequence[Tensor]:
                - dist1 (Tensor): Chamfer distance (xyz1 to xyz2) with
                    shape (B, N).
                - dist2 (Tensor): Chamfer distance (xyz2 to xyz1) with
                    shape (B, N).
                - idx1 (Tensor): Index of chamfer distance (xyz1 to xyz2)
                    with shape (B, N), which be used in compute gradient.
                - idx2 (Tensor): Index of chamfer distance (xyz2 to xyz2)
                    with shape (B, N), which be used in compute gradient.
        """
        batch_size, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        dist1 = jt.zeros(batch_size, n)
        dist2 = jt.zeros(batch_size, m)
        idx1 = jt.zeros((batch_size, n), dtype=jt.int32)
        idx2 = jt.zeros((batch_size, m), dtype=jt.int32)

        chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        self.save_for_backward = xyz1, xyz2, idx1, idx2                
        return dist1, dist2, idx1, idx2

    def grad(self,
                 grad_dist1,
                 grad_dist2,
                 grad_idx1=None,
                 grad_idx2=None):
        """
        Args:
            grad_dist1 (Tensor): Gradient of chamfer distance
                (xyz1 to xyz2) with shape (B, N).
            grad_dist2 (Tensor): Gradient of chamfer distance
                (xyz2 to xyz1) with shape (B, N).
        Returns:
            Tuple[Tensor, Tensor]:
            - grad_xyz1 (Tensor): Gradient of the point set with shape \
                (B, N, 2).
            - grad_xyz2 (Tensor):Gradient of the point set with shape \
                (B, N, 2).
        """
        xyz1, xyz2, idx1, idx2 = self.save_for_backward
        grad_dist1 = grad_dist1.contiguous()
        grad_dist2 = grad_dist2.contiguous()
        grad_xyz1 = jt.zeros(xyz1.size())
        grad_xyz2 = jt.zeros(xyz2.size())

        chamfer_distance_backward(xyz1, xyz2, idx1, idx2,
                                             grad_dist1, grad_dist2, grad_xyz1,
                                             grad_xyz2)
        return grad_xyz1, grad_xyz2


chamfer_distance = ChamferDistanceFunction.apply

def test():
    pointset1 = jt.array(
        [[[1.3, 9.39], [2.3, 9.39], [2.3, 10.39], [1.3, 10.39]],
         [[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]],
         [[1.6, 9.99], [2.3, 9.99], [2.3, 10.39], [1.6, 10.39]]])

    pointset2 = jt.array(
        [[[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]],
         [[1.3, 9.39], [2.3, 9.39], [2.3, 10.39], [1.3, 10.39]],
         [[1.0, 9.39], [3.0, 9.39], [3.0, 10.39], [1.0, 10.39]]])

    expected_dist1 = jt.array(
        [[0.0900, 0.4900, 0.4900, 0.0900], [0.0900, 0.4900, 0.4900, 0.0900],
         [0.5200, 0.6500, 0.4900, 0.3600]])
    expected_dist2 = jt.array(
        [[0.0900, 0.4900, 0.4900, 0.0900], [0.0900, 0.4900, 0.4900, 0.0900],
         [0.7200, 0.8500, 0.4900, 0.3600]])

    dist1, dist2, idx1, idx2 = chamfer_distance(pointset1, pointset2)
    np.testing.assert_allclose(dist1.numpy(),expected_dist1.numpy(),rtol=1e-2)
    np.testing.assert_allclose(dist2.numpy(),expected_dist2.numpy(),rtol=1e-2)

if __name__ == "__main__":
    jt.flags.use_cuda=1
    test()
