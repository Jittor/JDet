import jittor as jt 
ML_NMS_ROTATED_HEADER1 = r'''
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#undef out
#include <cassert>
#include <cmath>
#include <executor.h>
#include <algorithm>
#define CeilDIV(a,b) ((a+b-1)/b)
'''

ML_NMS_ROTATED_HEADER2 = r'''
namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
struct RotatedBox {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
  T x, y;
  HOST_DEVICE_INLINE Point(const T& px = 0, const T& py = 0) : x(px), y(py) {}
  HOST_DEVICE_INLINE Point operator+(const Point& p) const {
    return Point(x + p.x, y + p.y);
  }
  HOST_DEVICE_INLINE Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  HOST_DEVICE_INLINE Point operator-(const Point& p) const {
    return Point(x - p.x, y - p.y);
  }
  HOST_DEVICE_INLINE Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
HOST_DEVICE_INLINE T dot_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
HOST_DEVICE_INLINE T cross_2d(const Point<T>& A, const Point<T>& B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
HOST_DEVICE_INLINE void get_rotated_vertices(
    const RotatedBox<T>& box,
    Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  //double theta = box.a * 0.01745329251;
  //MODIFIED
  double theta = box.a;
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

template <typename T>
HOST_DEVICE_INLINE int get_intersection_points(
    const Point<T> (&pts1)[4],
    const Point<T> (&pts2)[4],
    Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0; // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) &&
          (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
HOST_DEVICE_INLINE int convex_hull_graham(
    const Point<T> (&p)[24],
    const int& num_in,
    Point<T> (&q)[24],
    bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto& start = p[t]; // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }
'''
ML_NMS_ROTATED_HEADER3 = r'''
// Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k; // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2; // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
HOST_DEVICE_INLINE T polygon_area(const Point<T> (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
HOST_DEVICE_INLINE T rotated_boxes_intersection(
    const RotatedBox<T>& box1,
    const RotatedBox<T>& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

} // namespace

template <typename T>
HOST_DEVICE_INLINE T
single_box_iou_rotated(T const* const box1_raw, T const* const box2_raw) {
  // we dont calculate IoU if two bboxes belong to two classes and set it to zero
  // box: [x,y,w,h,a,l]
  if (BOX_LENGTH==6 && box1_raw[5] != box2_raw[5])
    return 0.0;
  // shift center to the middle point to achieve higher precision in result
  RotatedBox<T> box1, box2;

  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];
  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

  const T area1 = box1.w * box1.h;
  const T area2 = box2.w * box2.h;
  if (area1 < 1e-14 || area2 < 1e-14) {
    return 0.f;
  }

  const T intersection = rotated_boxes_intersection<T>(box1, box2);
  const T iou = intersection / (area1 + area2 - intersection);
  return iou;
}
'''
ML_NMS_ROTATED_CPU_HEADER= ML_NMS_ROTATED_HEADER1+r'''
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
'''+ML_NMS_ROTATED_HEADER2+r'''
  // CPU version
  std::sort(
      q + 1, q + num_in, [](const Point<T>& A, const Point<T>& B) -> bool {
        T temp = cross_2d<T>(A, B);
        if (fabs(temp) < 1e-6) {
          return dot_2d<T>(A, A) < dot_2d<T>(B, B);
        } else {
          return temp > 0;
        }
      });
'''+ML_NMS_ROTATED_HEADER3

ML_NMS_ROTATED_CUDA_HEADER= ML_NMS_ROTATED_HEADER1+r'''
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
'''+ML_NMS_ROTATED_HEADER2+r'''
  // CUDA version
  // In the future, we can potentially use thrust
  // for sorting here to improve speed (though not guaranteed)
  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      T crossProduct = cross_2d<T>(q[i], q[j]);
      if ((crossProduct < -1e-6) ||
          (fabs(crossProduct) < 1e-6 && dist[i] > dist[j])) {
        auto q_tmp = q[i];
        q[i] = q[j];
        q[j] = q_tmp;
        auto dist_tmp = dist[i];
        dist[i] = dist[j];
        dist[j] = dist_tmp;
      }
    }
  }
'''+ML_NMS_ROTATED_HEADER3+r'''
template <typename T>
__global__ void nms_rotated_cuda_kernel(
    const int n_boxes,
    const float iou_threshold,
    const T* dev_boxes,
    unsigned long long* dev_mask) {
  // nms_rotated_cuda_kernel is modified from torchvision's nms_cuda_kernel

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // Compared to nms_cuda_kernel, where each box is represented with 4 values
  // (x1, y1, x2, y2), each rotated box is represented with 5 values
  // (x_center, y_center, width, height, angle_degrees) here.
  __shared__ T block_boxes[threadsPerBlock * BOX_LENGTH];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * BOX_LENGTH + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 0];
    block_boxes[threadIdx.x * BOX_LENGTH + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 1];
    block_boxes[threadIdx.x * BOX_LENGTH + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 2];
    block_boxes[threadIdx.x * BOX_LENGTH + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 3];
    block_boxes[threadIdx.x * BOX_LENGTH + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 4];
    if (BOX_LENGTH >= 6)
      block_boxes[threadIdx.x * BOX_LENGTH + 5] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * BOX_LENGTH + 5];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * BOX_LENGTH;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      // Instead of devIoU used by original horizontal nms, here
      // we use the single_box_iou_rotated function from box_iou_rotated_utils.h
      if (single_box_iou_rotated<T>(cur_box, block_boxes + i * BOX_LENGTH) >
          iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = CeilDIV(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
'''

ML_NMS_ROTATED_CPU_SRC=r'''
  @alias(dets,in0)
  @alias(order_t,in1)
  @alias(suppressed_t,in2)
  @alias(keep_t,out0)

  const int ndets = dets_shape0;
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
        continue;
      }

      auto ovr = single_box_iou_rotated(dets_p+i*BOX_LENGTH, dets_p+j*BOX_LENGTH);
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
      }
    }
  }
'''
ML_NMS_ROTATED_CUDA_SRC=r'''
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

  nms_rotated_cuda_kernel<<<blocks, threads, 0>>>(
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

def nms_rotated_cpu(dets,order_t,iou_threshold,box_length=6):
    suppressed_t = jt.zeros((dets.shape[0],)).uint8()

    keep = jt.code((dets.shape[0],),"bool",[dets,order_t,suppressed_t],
        cpu_header=f'''
        #define BOX_LENGTH {box_length}
        '''+ML_NMS_ROTATED_CPU_HEADER,
        cpu_src=f"const float iou_threshold = {iou_threshold};"
           +ML_NMS_ROTATED_CPU_SRC)
    return keep

def nms_rotated_cuda(dets,order_t,iou_threshold,box_length=6) :
    dets_sorted = dets[order_t,:]
    keep = jt.code((dets.shape[0],),"bool",[order_t,dets_sorted],
        cuda_header=f'''
        #define BOX_LENGTH {box_length}
        '''+ML_NMS_ROTATED_CUDA_HEADER,
        cuda_src=f"const float iou_threshold = {iou_threshold};"+ML_NMS_ROTATED_CUDA_SRC)
    return keep 

def ml_nms_rotated(dets,scores,labels,iou_threshold):
    assert dets.numel()>0 and dets.ndim==2
    assert dets.dtype==scores.dtype
    dets = jt.contrib.concat([dets,labels.cast(dets.dtype).unsqueeze(1)],dim=1)
    order_t,_ = scores.argsort(0,descending=True)

    if jt.flags.use_cuda:
        keep =  nms_rotated_cuda(dets,order_t,iou_threshold)
    else:
        keep =  nms_rotated_cpu(dets,order_t,iou_threshold)
    return jt.where(keep)[0]

def nms_rotated(dets,scores,iou_threshold):
    if dets.numel()==0:
      return jt.array([])
    assert dets.numel()>0 and dets.ndim==2
    assert dets.dtype==scores.dtype
    order_t,_ = scores.argsort(0,descending=True)

    if jt.flags.use_cuda:
        keep =  nms_rotated_cuda(dets,order_t,iou_threshold,box_length=5)
    else:
        keep =  nms_rotated_cpu(dets,order_t,iou_threshold,box_length=5)
    return jt.where(keep)[0]

def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_cfg,
                           max_num=-1,
                           score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand((multi_bboxes.shape[0], num_classes, 5))
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        return jt.zeros((0,6)), jt.zeros((0,)).int()
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    iou_thr = nms_cfg_.pop('iou_thr', 0.1)
    keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    inds,_ = scores.argsort(descending=True)

    if keep.size(0) > max_num:
        inds = inds[:max_num]
    bboxes = bboxes[inds]
    scores = scores[inds]
    labels = labels[inds]

    return jt.contrib.concat([bboxes, scores[:, None]], 1), labels
    
def test_ml_nms_rotated():
    dets = jt.array([[0,0,1,1,0],[0,0,0.5,0.5,0.3],[0,0,0.9,0.9,0]])
    scores = jt.array([0.1,0.2,0.3])
    labels = jt.array([1,1,1])
    print(ml_nms_rotated(dets,scores,labels,0.3))
    print(nms_rotated(dets,scores,0.3))

if __name__ == "__main__":
    test_ml_nms_rotated()

    jt.flags.use_cuda=1 
    test_ml_nms_rotated()
