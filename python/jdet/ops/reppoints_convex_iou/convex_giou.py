import os
import jittor as jt

CUDA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'convex_giou_kernel.cu')
CUDA_HEADER = open(CUDA_FILE_PATH, "r").read()

CUDA_SRC = r'''
@alias(pointsets,in0)
@alias(gtbboxes,in1)
@alias(grads,out0)

const int num_pointsets = pointsets_shape0;
const int num_gtbboxes = gtbboxes_shape0;

const int n_blocks = CeilDIV(num_pointsets, threadsPerBlock);
dim3 blocks(n_blocks);
dim3 threads(threadsPerBlock);

convex_giou_kernel<<<blocks, threads, 0>>>(
    num_pointsets,
    num_gtbboxes,
    pointsets_p,
    gtbboxes_p,
    grads_p
);
'''

# cuda only.
def reppoints_convex_giou(pointsets, gt_bboxes):
    assert pointsets.dtype == gt_bboxes.dtype
    assert len(pointsets.shape) == 2
    assert pointsets.shape[1] == 18
    assert len(gt_bboxes.shape) == 2
    assert gt_bboxes.shape[1] == 8
    assert gt_bboxes.shape[0] == pointsets.shape[0]

    num_pointsets = pointsets.shape[0]
    grad_iou = jt.code(
        shape=(num_pointsets, 19),
        dtype=pointsets.dtype,
        inputs=[pointsets, gt_bboxes],
        cuda_header=CUDA_HEADER,
        cuda_src=CUDA_SRC,
    )
    point_grad = grad_iou[:, :-1]
    iou = grad_iou[:, -1]
    return iou, point_grad
