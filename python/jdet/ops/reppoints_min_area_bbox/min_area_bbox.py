import os
import jittor as jt

CUDA_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'min_area_bbox.cu')
CUDA_HEADER = open(CUDA_FILE_PATH, "r").read()

CUDA_SRC=r'''
@alias(pointsets,in0)
@alias(bboxes,out0)
int num_pointsets = pointsets_shape0;
const int n_blocks = CeilDIV(num_pointsets, threadsPerBlock);

dim3 blocks(n_blocks);
dim3 threads(threadsPerBlock);
minareabbox_kernel<<<blocks, threads, 0>>>(
    num_pointsets,
    pointsets_p,
    bboxes_p
);
'''

def reppoints_min_area_bbox(pointsets):
    assert pointsets.shape[1] == 18
    if pointsets.shape[0] == 0:
        return jt.empty((0, 8))
    bboxes = jt.code(
        shape = (pointsets.shape[0], 8),
        dtype = pointsets.dtype,
        inputs = [pointsets],
        cuda_header = CUDA_HEADER,
        cuda_src = CUDA_SRC,
    )
    return bboxes

