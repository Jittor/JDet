import os
import jittor as jt

CUDA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'convex_iou_kernel.cu')
CUDA_HEADER = open(CUDA_FILE_PATH, "r").read()

CUDA_SRC=r'''
@alias(pointsets,in0)
@alias(gtbboxes,in1)
@alias(ious,out0)
int num_pointsets = pointsets_shape0;
int num_gtbboxes = gtbboxes_shape0;

if (num_pointsets > 0 && num_gtbboxes > 0) {
    const int n_blocks = CeilDIV(num_pointsets, threadsPerBlock);
    dim3 blocks(n_blocks);
    dim3 threads(threadsPerBlock);
    convex_iou_kernel<<<blocks, threads, 0>>>(
        num_pointsets,
        num_gtbboxes,
        pointsets_p,
        gtbboxes_p,
        ious_p
    );    
}
'''

# cuda only.
def reppoints_convex_iou(pointsets, gt_bboxes):
    assert pointsets.dtype == gt_bboxes.dtype
    assert len(pointsets.shape) == 2
    assert pointsets.shape[1] == 18
    assert len(gt_bboxes.shape) == 2
    assert gt_bboxes.shape[1] == 8

    num_pointsets = pointsets.shape[0]
    num_gtbboxes = gt_bboxes.shape[0]
    ious = jt.code(
        shape=(num_pointsets, num_gtbboxes),
        dtype=pointsets.dtype,
        inputs=[pointsets, gt_bboxes],
        cuda_header=CUDA_HEADER,
        cuda_src=CUDA_SRC,
    )
    return ious
