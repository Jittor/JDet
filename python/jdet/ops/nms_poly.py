import jittor as jt 
from shapely.geometry import Polygon

NMS_POLY_HEADER1 = r'''
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#undef out
#include <cassert>
#include <cmath>
#include <executor.h>
#include <algorithm>
#include<cstdio>
#include <vector>
#define CeilDIV(a,b) ((a+b-1)/b)
#define maxn 10
'''
NMS_POLY_HEADER2 = r'''
using namespace std;
int const threadsPerBlock = sizeof(unsigned long long) * 8;

const double eps=1E-8;

template <typename T>
HOST_DEVICE_INLINE int sig(T d){
    return(d>eps)-(d<-eps);
}

template <typename T>
struct Point{
    T x,y; Point(){}
    Point(T x,T y):x(x),y(y){}
    HOST_DEVICE_INLINE bool operator==(const Point<T>&p)const{
        return sig<T>(x-p.x)==0&&sig<T>(y-p.y)==0;
    }
};

template <typename T>
HOST_DEVICE_INLINE T cross(Point<T> o,Point<T> a,Point<T> b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

template <typename T>
HOST_DEVICE_INLINE T area(Point<T>* ps,int n){
    ps[n]=ps[0];
    T res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}

template <typename T>
HOST_DEVICE_INLINE int lineCross(Point<T> a,Point<T> b,Point<T> c,Point<T> d,Point<T>&p){
    T s1,s2;
    s1=cross<T>(a,b,c);
    s2=cross<T>(a,b,d);
    if(sig<T>(s1)==0&&sig<T>(s2)==0) return 2;
    if(sig<T>(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}

template <typename T>
HOST_DEVICE_INLINE void polygon_cut(Point<T>*p,int &n,Point<T> a,Point<T> b, Point<T>* pp){
//    static Point pp[maxn];
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig<T>(cross<T>(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig<T>(cross<T>(a,b,p[i]))!=sig<T>(cross<T>(a,b,p[i+1])))
            lineCross<T>(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(pp[i]==pp[i-1]))
            p[n++]=pp[i];
    while(n>1&&p[n-1]==p[0])n--;
}
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//

template <typename T>
HOST_DEVICE_INLINE T intersectArea(Point<T> a,Point<T> b,Point<T> c,Point<T> d){
    Point<T> o(0,0);
    int s1=sig<T>(cross<T>(o,a,b));
    int s2=sig<T>(cross<T>(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1==-1) swap(a,b);
    if(s2==-1) swap(c,d);
    Point<T> p[10]={o,a,b};
    int n=3;
    Point<T> pp[maxn];
    polygon_cut<T>(p,n,o,c, pp);
    polygon_cut<T>(p,n,c,d, pp);
    polygon_cut<T>(p,n,d,o, pp);
    
    T res=fabs(area<T>(p,n));
    if(s1*s2==-1) res=-res;
    return res;
}
//求两多边形的交面积
template <typename T>
HOST_DEVICE_INLINE T intersectArea(Point<T>*ps1,int n1,Point<T>*ps2,int n2){
    if(area<T>(ps1,n1)<0) reverse(ps1,ps1+n1);
    if(area<T>(ps2,n2)<0) reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    T res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea<T>(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}


template <typename T>
HOST_DEVICE_INLINE T iou_poly(const T * p, const T *q) {
    Point<T> ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }

    T inter_area = intersectArea<T>(ps1, n1, ps2, n2);
    T union_area = fabs(area<T>(ps1, n1)) + fabs(area<T>(ps2, n2)) - inter_area;
    T iou = inter_area / union_area;

    return iou;
}
'''

NMS_POLY_CPU_HEADER= NMS_POLY_HEADER1+r'''
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
'''+NMS_POLY_HEADER2

NMS_POLY_CUDA_HEADER= NMS_POLY_HEADER1+r'''
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
'''+NMS_POLY_HEADER2+r'''
template <typename T>
__global__ void nms_poly_cuda_kernel(
    const int n_boxes,
    const float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  // nms_poly_cuda_kernel is modified from torchvision's nms_cuda_kernel

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // (x1, y1, x2, y2,x1, y1, x2, y2), each ploy is represented with 8 values
  __shared__ T block_boxes[threadsPerBlock * 8];
  if (threadIdx.x < col_size) {
    for(int i=0;i<8;i++){
        block_boxes[threadIdx.x * 8 + i] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 8 + i];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 8;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      // Instead of devIoU used by original horizontal nms, here
      if (iou_poly<T>(cur_box, block_boxes + i * 8) >
          iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = CeilDIV(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''

NMS_POLY_CPU_SRC=r'''
  @alias(dets,in0)
  @alias(order_t,in1)
  @alias(suppressed_t,in2)
  @alias(keep_t,out0)

  const int ndets = dets_shape0;
  const int nps = dets_shape1;

  auto suppressed = suppressed_t_p;
  auto keep = keep_t_p;
  auto order = order_t_p;

  memset(keep,0,keep_t->size);

  int num_to_keep = 0;

  for (int _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) {
      continue;
    }

    keep[i] = true;

    for (int _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) {
          printf("fuxk\n");
        continue;
      }

      auto ovr = iou_poly(dets_p+i*nps, dets_p+j*nps);
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
'''
NMS_POLY_CUDA_SRC=r'''
  @alias(order,in0)
  @alias(dets_sorted,in1)
  @alias(keep,out0)

  cudaMemsetAsync(keep_p,0,keep->size);

  const int dets_num = dets_sorted_shape0;


  const int col_blocks = CeilDIV(dets_num, threadsPerBlock);

  int matrices_size = dets_num * col_blocks * sizeof(unsigned long long);
  size_t mask_allocation;
  unsigned long long* mask_p = (unsigned long long*)exe.allocator->alloc(matrices_size, mask_allocation);

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  nms_poly_cuda_kernel<<<blocks, threads, 0>>>(
            dets_num,
            iou_threshold,
            dets_sorted_p,
            mask_p);

  checkCudaErrors(cudaDeviceSynchronize());

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  for (int i = 0; i < dets_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_p[order_p[i]] = true;
      unsigned long long* p = mask_p + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
exe.allocator->free(mask_p, matrices_size, mask_allocation);
'''

def nms_poly_cpu(dets,order_t,iou_threshold):
    suppressed_t = jt.zeros((dets.shape[0],)).uint8()

    keep = jt.code((dets.shape[0],),"bool",[dets,order_t,suppressed_t],cpu_header=NMS_POLY_CPU_HEADER,
        cpu_src=f"const float iou_threshold = {iou_threshold};"+NMS_POLY_CPU_SRC)
    return keep

def nms_poly_cuda(dets,order_t,iou_threshold) :
    dets_sorted = dets[order_t,:]
    keep = jt.code((dets.shape[0],),"bool",[order_t,dets_sorted],cuda_header=NMS_POLY_CUDA_HEADER,
        cuda_src=f"const float iou_threshold = {iou_threshold};"+NMS_POLY_CUDA_SRC)
    return keep 

def nms_poly(dets,scores,iou_threshold):
    assert dets.numel()>0 and dets.ndim==2
    assert dets.dtype==scores.dtype
    order_t,_ = scores.argsort(0,descending=True)

    if jt.flags.use_cuda:
        keep =  nms_poly_cuda(dets,order_t,iou_threshold)
    else:
        keep =  nms_poly_cpu(dets,order_t,iou_threshold)
    return jt.where(keep)[0]


def iou_poly(poly1,poly2):
    poly1 = Polygon(poly1.reshape(4,2))
    poly2 = Polygon(poly2.reshape(4,2))
    inter_area = poly1.intersection(poly2).area
    iou = inter_area/(poly1.area+poly2.area-inter_area)
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


def test_nms_poly():
    dets = jt.array([[0,0,1,0,1,1,0,1],[0,0,0.3,0.2,0.5,0.5,0.3,0.7],[0,0,0.9,0,0.9,0.9,0,0.9]])
    scores = jt.array([0.1,0.2,0.3])
    print(nms_poly(dets,scores,0.2))

if __name__ == "__main__":
    test_nms_poly()

    jt.flags.use_cuda=1 
    test_nms_poly()
