import jittor as jt
from jittor import nn
from jdet.models.utils.weight_init import normal_init

HEADER=r"""
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return std::min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t* bottom_data,
    const int height, const int width,
    scalar_t y, scalar_t x) {
    // deal with cases that inverse elements are out of feature map boundary
    // if the feature map's size is [height, width], then its valid pixel
    // coordinates range is: x in [0, width-1] y in [0, height-1]
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
    }
    else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (scalar_t)x_low;
    }
    else {
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
__device__ void bilinear_interpolate_gradient(const int height, const int width,
    scalar_t y, scalar_t x,
    scalar_t& w1, scalar_t& w2,
    scalar_t& w3, scalar_t& w4,
    int& x_low, int& x_high,
    int& y_low, int& y_high) {
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
    }
    else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (scalar_t)x_low;
    }
    else {
        x_high = x_low + 1;
    }

    scalar_t ly = y - y_low;
    scalar_t lx = x - x_low;
    scalar_t hy = 1. - ly;
    scalar_t hx = 1. - lx;

    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename scalar_t>
__global__ void feature_refine_forward_kernel(
    const int nthreads,
    const int points,
    const scalar_t* bottom_data,
    const scalar_t* best_bboxes,  // of shape (n, h, w, 5)
    const float spatial_scale, const int channels, const int height,
    const int width, scalar_t* top_data) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, h, w) is an element in the aligned output
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height /
            channels;  // refers to the n^th image within a minibatch

        const scalar_t* bbox_offset =
            best_bboxes + ((n * height + h) * width + w) * 5;
        // for rbbox, there are 5 entries: [x_ctr, y_ctr, w, h, ang]
        scalar_t roi_y = bbox_offset[0] * spatial_scale;
        scalar_t roi_x = bbox_offset[1] * spatial_scale;

        scalar_t px[5] = { roi_x, 0, 0, 0, 0 };
        scalar_t py[5] = { roi_y, 0, 0, 0, 0 };

        if (points > 1) {
            scalar_t roi_w = bbox_offset[2] * spatial_scale;
            scalar_t roi_h = bbox_offset[3] * spatial_scale;
            scalar_t roi_a = bbox_offset[4];

            scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
            scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
            scalar_t wx = cosa * w_2, wy = sina * w_2;
            scalar_t hx = -sina * h_2, hy = cosa * h_2;

            px[1] = roi_x + wx + hx; py[1] = roi_y + wy + hy;
            px[2] = roi_x - wx + hx; py[2] = roi_y - wy + hy;
            px[3] = roi_x - wx - hx; py[3] = roi_y - wy - hy;
            px[4] = roi_x + wx - hx; py[4] = roi_y + wy - hy;
        }

        const scalar_t* offset_bottom_data =
            bottom_data + (n * channels + c) * height * width;

        scalar_t output_val = bottom_data[index];
        for (int i = 0; i < points; i++) {
            output_val += bilinear_interpolate<scalar_t>(offset_bottom_data, height,
                width, py[i], px[i]);
        }
        top_data[index] = output_val;
    }
}

template <typename scalar_t>
__global__ void feature_refine_backward_kernel(
    const int nthreads,
    const int points,
    const scalar_t* top_diff,
    const scalar_t* best_bboxes,  // of shape (n, h, w, 5)
    const float spatial_scale, const int channels, const int height,
    const int width, scalar_t* bottom_diff) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        // (n, c, h, w) is an element in the input diff
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height /
            channels;  // refers to the n^th image within a minibatch

        const scalar_t* bbox_offset =
            best_bboxes + ((n * height + h) * width + w) * 5;
        // for rbbox, there are 5 entries: [x_ctr, y_ctr, w, h, ang]
        scalar_t roi_y = bbox_offset[0] * spatial_scale;
        scalar_t roi_x = bbox_offset[1] * spatial_scale;

        scalar_t px[5] = { roi_x, 0, 0, 0, 0 };
        scalar_t py[5] = { roi_y, 0, 0, 0, 0 };

        if (points > 1) {
            scalar_t roi_w = bbox_offset[2] * spatial_scale;
            scalar_t roi_h = bbox_offset[3] * spatial_scale;
            scalar_t roi_a = bbox_offset[4];

            scalar_t w_2 = roi_w / 2, h_2 = roi_h / 2;
            scalar_t cosa = cosf(roi_a), sina = sinf(roi_a);
            scalar_t wx = cosa * w_2, wy = sina * w_2;
            scalar_t hx = -sina * h_2, hy = cosa * h_2;

            px[1] = roi_x + wx + hx; py[1] = roi_y + wy + hy;
            px[2] = roi_x - wx + hx; py[2] = roi_y - wy + hy;
            px[3] = roi_x - wx - hx; py[3] = roi_y - wy - hy;
            px[4] = roi_x + wx - hx; py[4] = roi_y + wy - hy;
        }

        scalar_t* offset_bottom_diff =
            bottom_diff + (n * channels + c) * height * width;
        scalar_t value_top_diff = top_diff[index];

        atomicAdd(bottom_diff + index, value_top_diff);
        for (int i = 0; i < points; i++) {
            scalar_t w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            bilinear_interpolate_gradient<scalar_t>(height, width, py[i], px[i], w1,
                w2, w3, w4, x_low, x_high, y_low,
                y_high);
            scalar_t g1 = value_top_diff * w1;
            scalar_t g2 = value_top_diff * w2;
            scalar_t g3 = value_top_diff * w3;
            scalar_t g4 = value_top_diff * w4;
            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
            }
        }
    }
}
"""
def feature_refine_forward(features,best_bboxes,spatial_scale,points):
    src = f"""
    const int output_size = {features.numel()};
    feature_refine_forward_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK >>>(
                    output_size, {points}, in0_p, in1_p, {spatial_scale},in0_shape1, in0_shape2, in0_shape3, out0_p);
    """
    return jt.code(features.shape,features.dtype,[features,best_bboxes],cuda_header=HEADER,cuda_src=src)

    

def feature_refine_backward(top_grad,best_bboxes,spatial_scale,points):
    bottom_grad = jt.zeros_like(top_grad)
    src=f"""
    const int output_size = {top_grad.numel()};
    feature_refine_backward_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK >>> (
                    output_size, {points}, in0_p, in1_p, {spatial_scale},
                    in0_shape1, in0_shape2, in0_shape3,out0_p);
    """
    return jt.code(top_grad.shape,top_grad.dtype,[top_grad,best_bboxes],cuda_header=HEADER,cuda_src=src)
    

class FeatureRefineFunction(jt.Function):

    def execute(self, features, best_rbboxes, spatial_scale, points=1):
        self.spatial_scale = spatial_scale
        self.points = points
        self.best_rbboxes = best_rbboxes
        assert points in [1, 5] 
        output = feature_refine_forward(features, best_rbboxes, spatial_scale, points)
        return output

    def grad(self, grad_output):
        best_rbboxes = self.best_rbboxes
        points = self.points
        spatial_scale = self.spatial_scale
        grad_input = feature_refine_backward(grad_output, best_rbboxes, spatial_scale, points)
        return grad_input, None, None, None


feature_refine = FeatureRefineFunction.apply

class FR(nn.Module):
    def __init__(self, spatial_scale, points=1):
        super(FR, self).__init__()
        self.spatial_scale = float(spatial_scale)
        self.points = points

    def execute(self, features, best_rbboxes):
        return feature_refine(features, best_rbboxes, self.spatial_scale, self.points)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(spatial_scale={}, points={})'.format(self.spatial_scale, self.points)
        return format_str



class FeatureRefineModule(nn.Module):
    def __init__(
            self,
            in_channels,
            featmap_strides,
            conv_cfg=None,
            norm_cfg=None):
        super(FeatureRefineModule, self).__init__()
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.fr = nn.ModuleList([FR(spatial_scale=1 / s)
                                 for s in self.featmap_strides])
        self.conv_5_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(5, 1),
            stride=1,
            padding=(2, 0))
        self.conv_1_5 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=(1, 5),
            stride=1,
            padding=(0, 2))
        self.conv_1_1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1)

    def init_weights(self):
        normal_init(self.conv_5_1, std=0.01)
        normal_init(self.conv_1_5, std=0.01)
        normal_init(self.conv_1_1, std=0.01)

    def execute(self, x, best_rbboxes):
        """
        Args:
            x (list[Tensor]):
                feature maps of multiple scales
            best_rbboxes (list[list[Tensor]]):
                best rbboxes of multiple scales of multiple images
        """
        mlvl_rbboxes = [jt.concat(best_rbbox) for best_rbbox in zip(*best_rbboxes)]
        out = []
        for x_scale, best_rbboxes_scale, fr_scale in zip(x, mlvl_rbboxes, self.fr):
            feat_scale_1 = self.conv_5_1(self.conv_1_5(x_scale))
            feat_scale_2 = self.conv_1_1(x_scale)
            feat_scale = feat_scale_1 + feat_scale_2
            feat_refined_scale = fr_scale(feat_scale, best_rbboxes_scale)
            out.append(x_scale + feat_refined_scale)
        return out

def test():
    import math
    feat_size_x = 32
    feat_size_y = 32
    stride = 8.0
    spatial_scale = 1.0 / stride
    base_size = 4.0 * stride
    num_imgs = 2
    num_chns = 16

    feat = jt.randn((num_imgs, num_chns, feat_size_x, feat_size_y))

    xc, yc = jt.meshgrid(stride * jt.arange(feat_size_x), stride * jt.arange(feat_size_y))
    xc = xc[None,:,:]
    yc = yc[None,:,:]
    xc = xc + base_size * jt.randn(num_imgs, feat_size_x, feat_size_y)
    yc = yc + base_size * jt.randn(num_imgs, feat_size_x, feat_size_y)
    w = base_size * jt.randn(num_imgs, feat_size_x, feat_size_y).exp()
    h = base_size * jt.randn(num_imgs, feat_size_x, feat_size_y).exp()
    a = -math.pi / 2 * jt.rand(num_imgs, feat_size_x, feat_size_y)
    bbbox = jt.stack([xc, yc, w, h, a], dim=-1)

    f1 = FR(spatial_scale, points=1)
    f2 = FR(spatial_scale, points=5)

    o1 = f1(feat,bbbox)
    o2 = f2(feat,bbbox)

    print(o1.sum(),o2.sum())
    print(jt.grad(o1+o2,feat).sum())


if __name__ == "__main__":
    jt.flags.use_cuda=1
    test()


