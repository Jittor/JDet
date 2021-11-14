from jittor import Function
import jittor as jt

CUDA_HEAD = r'''
template <typename T>
__global__ void convex_sort_kernel(
    const int nbs, const int npts, const int index_size, const bool circular,
    const T* x, const T* y, const T* m, const int* start_index,
    const int* order, int* convex_index) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nbs) {
    const T* sub_x = x + i * npts;
    const T* sub_y = y + i * npts;
    const T* sub_m = m + i * npts;
    const int* sub_order = order + i * npts;
    const int sub_start_index = start_index[i];

    int* sub_convex_index = convex_index + i * index_size;
    sub_convex_index[0] = sub_start_index;
    int c_i = 0;

    for (int _j = 0; _j < npts; _j++) {
      const int j = sub_order[_j];
      if (j == sub_start_index)continue;
      if (sub_m[j] < 0.5)continue;

      const T x0 = sub_x[j];
      const T y0 = sub_y[j];
      T x1 = sub_x[sub_convex_index[c_i]];
      T y1 = sub_y[sub_convex_index[c_i]];
      T d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
      if (d < 0.000001)continue;

      if (c_i < 2) {
	sub_convex_index[++c_i] = j;
      }
      else {
	T x2 = sub_x[sub_convex_index[c_i - 1]];
	T y2 = sub_y[sub_convex_index[c_i - 1]];
	while(1) {
	  T t = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
	  if (t >= 0) {
	    sub_convex_index[++c_i] = j;
	    break;
	  }
	  else {
	    if (c_i <= 1) {
	      sub_convex_index[c_i] = j;
	      break;
	    }
	    else {
	      c_i--;
	      x1 = sub_x[sub_convex_index[c_i]];
	      y1 = sub_y[sub_convex_index[c_i]];
	      x2 = sub_x[sub_convex_index[c_i - 1]];
	      y2 = sub_y[sub_convex_index[c_i - 1]];
	    }
	  }
	}
      }
    }
    if (circular) sub_convex_index[++c_i] = sub_convex_index[0];
  }
}
'''

def convex_sort_cpu(pts,masks,circular):
    INF = 10000000
    nbs = pts.size(0)
    npts = pts.size(1)
    index_size = npts+1 if circular else npts

    masks_t = masks.cast(pts.dtype)
    x_t = pts[:,:,0]
    y_t = pts[:,:,1]

    masked_y = masks_t * y_t + (1 - masks_t) * INF
    start_index_t,_ = masked_y.argmin(1, keepdims=True)
    start_x = x_t.gather(1, start_index_t)
    start_y = y_t.gather(1, start_index_t)

    pts_cos = (x_t - start_x) / jt.sqrt((x_t - start_x)*(x_t - start_x) + (y_t - start_y)*(y_t - start_y) + EPS)
    order_t,_ = pts_cos.argsort(1, descending=True)

    convex_index_t = jt.full((nbs, index_size), -1).int()
    if npts == 0:
        return convex_index_t

    SRC=f"""
    int nbs = {nbs};
    int npts = {npts};
    int index_size = {index_size};
    bool circular = {"true" if circular else "false"};
    """+r"""
    auto x = in0_p;
    auto y = in1_p;
    auto m = in2_p;
    auto start_index = in3_p;
    auto order = in4_p;
    auto convex_index = out0_p;

    for (int i = 0; i < nbs; i++) {
        auto* sub_x = x + i * npts;
        auto* sub_y = y + i * npts;
        auto* sub_m = m + i * npts;
        int* sub_order = order + i * npts;
        int* sub_convex_index = convex_index + i * index_size;
        int sub_start_index = start_index[i];

        sub_convex_index[0] = sub_start_index;
        int c_i = 0;

        for (int _j = 0; _j < npts; _j++) {
            int j = sub_order[_j];
            if (j == sub_start_index) continue;
            if (sub_m[j] < 0.5) continue;

            auto x0 = sub_x[j];
            auto y0 = sub_y[j];
            auto x1 = sub_x[sub_convex_index[c_i]];
            auto y1 = sub_y[sub_convex_index[c_i]];
            auto d = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
            if (d < 0.000001)continue;

            if (c_i < 2) {
                sub_convex_index[++c_i] = j;
            }
            else {
                auto x2 = sub_x[sub_convex_index[c_i - 1]];
                auto y2 = sub_y[sub_convex_index[c_i - 1]];
                    while(1) {
                    auto t = (x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2);
                    if (t >= 0) {
                        sub_convex_index[++c_i] = j;
                        break;
                    }
                    else {
                    if (c_i <= 1) {
                        sub_convex_index[c_i] = j;
                        break;
                    }else {
                        c_i--;
                        x1 = sub_x[sub_convex_index[c_i]];
                        y1 = sub_y[sub_convex_index[c_i]];
                        x2 = sub_x[sub_convex_index[c_i - 1]];
                        y2 = sub_y[sub_convex_index[c_i - 1]];
                    }
                }
                }
            }
        }
        if (circular)sub_convex_index[++c_i] = sub_convex_index[0];
    }
    """
    convex_index_t = jt.code(outputs=[convex_index_t],inputs=[x_t,y_t,masks_t,start_index_t,order_t],cpu_src=SRC)
    return convex_index_t


def convex_sort_gpu(pts,masks,circular):
    INF = 10000000
    EPS = 0.000001
    nbs = pts.size(0)
    npts = pts.size(1)
    index_size = npts+1 if circular else npts

    masks_t = masks.cast(pts.dtype)
    x_t = pts[:,:,0]
    y_t = pts[:,:,1]

    masked_y = masks_t * y_t + (1 - masks_t) * INF
    start_index_t,_ = masked_y.argmin(1, keepdims=True)
    start_x = x_t.gather(1, start_index_t)
    start_y = y_t.gather(1, start_index_t)

    pts_cos = (x_t - start_x) / jt.sqrt((x_t - start_x)*(x_t - start_x) + (y_t - start_y)*(y_t - start_y) + EPS)
    order_t,_ = pts_cos.argsort(1, descending=True)

    convex_index_t = jt.full((nbs, index_size), -1).int()
    if npts == 0:
        return convex_index_t

    SRC = f"""
    const int nbs = out0_shape0;
    const int index_size = out0_shape1; 
    const bool circular = {"true" if circular else "false"};
    const int npts = circular? index_size-1:index_size;
    dim3 blocks((nbs+511)/512);
    dim3 threads(512);
    convex_sort_kernel<<<blocks, threads>>>(nbs, npts, index_size, circular,in0_p, in1_p, in2_p,in3_p, in4_p,out0_p);
    """
    jt.code(outputs=[convex_index_t],inputs=[x_t,y_t,masks_t,start_index_t,order_t],cuda_header=CUDA_HEAD,cuda_src=SRC)
    return convex_index_t



def convex_sort(pts, masks, circular=True):
    assert pts.size(0) == masks.size(0) and pts.size(1) == masks.size(1) 
    if jt.flags.use_cuda:
        return convex_sort_gpu(pts,masks,circular)
    return convex_sort_cpu(pts,masks,circular)


def test_convex_sort():
    # pts = jt.array([])
    # masks = jt.array([])

    convex_index = convex_sort(pts,masks)
    print(convex_index)

if __name__ == "__main__":
    test_convex_sort()
    jt.flags.use_cuda = 1
    test_convex_sort()



