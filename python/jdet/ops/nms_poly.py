import jittor as jt 
from shapely.geometry import Polygon

HEADER = r'''
#undef out
#include <executor.h>
#include <vector>
#include <iostream>
#define THCCeilDiv(a,b) ((a + b - 1) / b)
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


#define maxn 10
const double eps=1E-8;

__device__ inline int sig(float d){
    return(d>1e-8)-(d<-1e-8);
}

__device__ inline int point_eq(const float2 a, const float2 b) {
    return sig(a.x - b.x) == 0 && sig(a.y - b.y)==0;
}

__device__ inline void point_swap(float2 *a, float2 *b) {
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2* last)
{
    while ((first!=last)&&(first!=--last)) {
        point_swap (first,last);
        ++first;
    }
}

__device__ inline float cross(float2 o,float2 a,float2 b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
__device__ inline float area(float2* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(float2 a,float2 b,float2 c,float2 d,float2&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}

__device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b, float2* pp){

    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(point_eq(pp[i], pp[i-1])))
            p[n++]=pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while(n>1&&point_eq(p[n-1], p[0]))n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a,float2 b,float2 c,float2 d){
    float2 o = make_float2(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    // if(s1==-1) swap(a,b);
    // if(s2==-1) swap(c,d);
    if (s1 == -1) point_swap(&a, &b);
    if (s2 == -1) point_swap(&c, &d);
    float2 p[10]={o,a,b};
    int n=3;
    float2 pp[maxn];
    polygon_cut(p,n,o,c,pp);
    polygon_cut(p,n,c,d,pp);
    polygon_cut(p,n,d,o,pp);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
//求两多边形的交面积
__device__ inline float intersectArea(float2*ps1,int n1,float2*ps2,int n2){
    if(area(ps1,n1)<0) point_reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) point_reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}

// TODO: optimal if by first calculate the iou between two hbbs
__device__ inline float devPolyIoU(float const * const p, float const * const q) {
    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }
    return iou;
}

__global__ void poly_nms_kernel(const int n_polys, const float nms_overlap_thresh,
                            const float *dev_polys, unsigned long long *dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size =
            min(n_polys - row_start * threadsPerBlock, threadsPerBlock);
    const int cols_size =
            min(n_polys - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_polys[threadsPerBlock * 9];
    if (threadIdx.x < cols_size) {
        block_polys[threadIdx.x * 9 + 0] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
        block_polys[threadIdx.x * 9 + 1] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
        block_polys[threadIdx.x * 9 + 2] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
        block_polys[threadIdx.x * 9 + 3] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
        block_polys[threadIdx.x * 9 + 4] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
        block_polys[threadIdx.x * 9 + 5] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
        block_polys[threadIdx.x * 9 + 6] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
        block_polys[threadIdx.x * 9 + 7] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
        block_polys[threadIdx.x * 9 + 8] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_polys + cur_box_idx * 9;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < cols_size; i++) {
            if (devPolyIoU(cur_box, block_polys + i * 9) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = THCCeilDiv(n_polys, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}
'''
def poly_nms(boxes,nms_overlap_thresh):
    assert boxes.ndim==2 and boxes.shape[1]==9
    assert jt.flags.use_cuda 
    scores = boxes[:,8]
    order_t,_ = scores.argsort(0,descending=True)
    boxes_sorted = boxes[order_t]
    SRC =f"""
    const float nms_overlap_thresh = {nms_overlap_thresh};
    """+r"""
    @alias(boxes_sorted,in0)
    @alias(keep,out0)
    cudaMemsetAsync(keep_p,0,keep->size);
    const int boxes_num = boxes_sorted_shape0;
    const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

    int matrices_size = boxes_num * col_blocks * sizeof(unsigned long long);
    size_t mask_allocation;
    unsigned long long* mask_p = (unsigned long long*)exe.allocator->alloc(matrices_size, mask_allocation);
    dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
                THCCeilDiv(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    poly_nms_kernel<<<blocks, threads, 0>>>(boxes_num,
                                        nms_overlap_thresh,
                                        boxes_sorted_p,
                                        mask_p);
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<unsigned long long> remv(col_blocks);

    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_p[i] = true;
            unsigned long long* p = mask_p + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }
    exe.allocator->free(mask_p, matrices_size, mask_allocation);
    """
    keep = jt.code((boxes.shape[0],),"bool",[boxes_sorted],cuda_header=HEADER,cuda_src=SRC)
    return order_t[keep] 

def multiclass_poly_nms(bboxes, scores, labels, thresh):
    max_coordinate = bboxes.max() - bboxes.min()
    offsets = labels * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    
    keep = poly_nms(jt.concat([bboxes_for_nms,scores[:,None]],dim=1),thresh)

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    dets = jt.concat([bboxes,scores[:,None]],dim=1)
    return dets,labels

def iou_poly(poly1,poly2):
    poly1 = Polygon(poly1.reshape(4,2))
    poly2 = Polygon(poly2.reshape(4,2))
    inter_area = poly1.intersection(poly2).area
    iou = inter_area/max(poly1.area+poly2.area-inter_area,0.01)
    return iou

def nms_poly_numpy(dets, thresh):
    scores = dets[:, 8]
    polys = dets[:,:8]
    areas = []
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def test2(i):
    import numpy as np
    import pickle

    bboxes, scores, labels = pickle.load(open(f"/home/lxl/diff/poly_nms_in_{i}.pkl","rb"))
    bboxes_j, scores_j, keep_j = multiclass_poly_nms(jt.array(bboxes),jt.array(scores),jt.array(labels),0.1)
    

    bboxes_t, scores_t, keep_t = pickle.load(open(f"/home/lxl/diff/poly_nms_out_{i}.pkl","rb"))
    # print(keep_t)
    # print(keep_j)

    # print(keep_j,np.sort(keep_t))
    # np.testing.assert_allclose(bboxes_j.numpy(),bboxes_t,rtol=1e-05)
    # np.testing.assert_allclose(scores_j.numpy(),scores_t,rtol=1e-05)
    # np.testing.assert_allclose(keep_j.numpy(),keep_t,rtol=1e-05)

    if keep_j.numpy().shape!=keep_t.shape or not np.allclose(keep_j.numpy(),keep_t,rtol=1e-05):
        print(i,keep_j.shape,keep_t.shape)
        print(keep_j.numpy())
        print(keep_t)

if __name__ == "__main__":
    jt.flags.use_cuda=1 
    # test2(19)
    # for i in range(1000):
    #     test2(i)




