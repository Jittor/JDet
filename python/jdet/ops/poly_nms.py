import numpy as np
from shapely.geometry import Polygon

# TODO
# use code_op implement iou_poly
def iou_poly(poly1,poly2):
    poly1 = Polygon(poly1.reshape(4,2))
    poly2 = Polygon(poly2.reshape(4,2))
    inter_area = poly1.intersection(poly2).area
    iou = inter_area/(poly1.area+poly2.area-inter_area)
    return iou

def poly_nms(dets, thresh):
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


def test():
    poly1 = np.array([[0,0],[1,1],[0,2],[-1,1]])
    poly2 = np.array([[1,0],[2,1],[1,2],[0,1]])
    poly2 = np.array([[10,0],[11,1],[10,2],[9,1]])
    poly2 = np.array([[0,0],[1,1],[0,2],[-1,1]])
    print(iou_poly(poly1,poly2))




if __name__ == "__main__":
    test()