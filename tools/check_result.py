import argparse
from matplotlib.pyplot import cla
import numpy as np
from shapely.geometry import Polygon
from xml.dom.minidom import parse
from tqdm import tqdm
import os

FAIR2M_CLASSES = ['Airplane', 'Ship', 'Vehicle', 'Basketball_Court', 'Tennis_Court', 
        "Football_Field", "Baseball_Field", 'Intersection', 'Roundabout', 'Bridge']

def iou_poly(poly1,poly2):
    poly1 = Polygon(poly1.reshape(4,2))
    poly2 = Polygon(poly2.reshape(4,2))
    inter_area = poly1.intersection(poly2).area
    iou = inter_area/max(poly1.area+poly2.area-inter_area,0.01)
    return iou

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_dota(dets,gts,iou_func,ovthresh=0.5,use_07_metric=False):
    dets = np.array(dets.tolist())
    npos = sum([sum(~gts[k]["difficult"]) for k in gts])
    nd = len(dets)
    if nd==0 or npos==0:
        return 0.,0.,0.

    confidence = dets[:,-1]
    dets = dets[:,:-1]

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    scores = confidence[sorted_ind]

    ## note the usage only in numpy not for list
    dets = dets[sorted_ind, :]
    # go down dets and mark TPs and FPs
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d,det in enumerate(dets):
        bb = det[1:].astype(float)
        ovmax = -np.inf
        R = gts[int(det[0])]
        BBGT = R["box"].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = iou_func(BBGT_keep[index], bb)
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)

    # print('npos num:', npos)
    # print("n dets",nd)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def evaluate(dets, gts, classes):
    # dets: [
    # {
    #   name: "1.tif",
    #   bboxes: [{label:label, poly:poly, score:score}, ...],
    # }, ...
    # ]
    # gts: [
    # {
    #   name: "1.tif",
    #   bboxes: [{label:label, poly:poly}, ...],
    # }, ...
    # ]    # class: ["classname",...]
    img2idx = dict()
    dets_dict = dict()
    gts_dict = dict()
    for idx, name in enumerate(classes):
        dets_dict[name] = []
        gts_dict[name] = {}
    for idx, gt in enumerate(gts):
        img2idx[gt["name"]] = idx
        bboxes = gt["bboxes"]
        name_array = np.array([idx], dtype=np.float32)
        single_gt_dict = {}
        for classname in classes:
            single_gt_dict[classname] = {}
            single_gt_dict[classname]["box"] = []
            single_gt_dict[classname]["det"] = []
            single_gt_dict[classname]["difficult"] = []
        for bbox in bboxes:
            label = bbox["label"]
            single_gt_dict[label]["box"].append(bbox["poly"])
            single_gt_dict[label]["det"].append(False)
            single_gt_dict[label]["difficult"].append(False)
        for classname in classes:
            if len(single_gt_dict[classname]["box"]) == 0:
                single_gt_dict[classname]["box"] = np.zeros((0,8)).astype(float)
                single_gt_dict[classname]["det"] = np.zeros((0)).astype(bool)
                single_gt_dict[classname]["difficult"] = np.zeros((0)).astype(bool)
            else:
                single_gt_dict[classname]["box"] = np.stack(single_gt_dict[classname]["box"])
                single_gt_dict[classname]["det"] = np.array(single_gt_dict[classname]["det"])
                single_gt_dict[classname]["difficult"] = np.array(single_gt_dict[classname]["difficult"])
            gts_dict[classname][idx] = single_gt_dict[classname]
    for det in dets:
        nameidx = img2idx[det["name"]]
        bboxes = det["bboxes"]
        name_array = np.array([nameidx], dtype=np.float32)
        for bbox in bboxes:
            score = bbox["score"] if "score" in bbox.keys() else 0.
            score_array = np.array([score], dtype=np.float32)
            new_det = np.concatenate([name_array, bbox["poly"], score_array])
            dets_dict[bbox["label"]].append(new_det)
    aps = {}
    for classname in tqdm(classes):
        if len(dets_dict[classname]) == 0:
            cls_dets = np.zeros((0, 10), dtype=np.float32)
        else:
            cls_dets = np.stack(dets_dict[classname])
        rec, prec, ap = voc_eval_dota(cls_dets, gts_dict[classname], iou_func=iou_poly)
        aps["eval/"+classname+"_AP"]=ap 
    map = sum(list(aps.values()))/len(aps)
    aps["eval/0_meanAP"]=map
    return aps

def parse_file(src, read_score=False):
    domTree = parse(src)
    rootNode = domTree.documentElement
    objects = rootNode.getElementsByTagName("objects")[0].getElementsByTagName("object")
    imgname = rootNode.getElementsByTagName("source")[0].getElementsByTagName("filename")[0].childNodes[0].data
    box_list = []
    for obj in objects:
        name=obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("name")[0].childNodes[0].data
        points=obj.getElementsByTagName("points")[0].getElementsByTagName("point")
        bbox=[]
        for point in points[:4]:
            x=point.childNodes[0].data.split(",")[0]
            y=point.childNodes[0].data.split(",")[1]
            bbox.append(float(x))
            bbox.append(float(y))
        name = name.replace(" ", "_")
        box_dict = {"label":name, "poly":bbox}
        if read_score:
            score = obj.getElementsByTagName("possibleresult")[0].getElementsByTagName("probability")[0].childNodes[0].data
            box_dict["score"] = float(score)
        box_list.append(box_dict)
    return dict(
        name=imgname,
        bboxes=box_list,
    )

def parse_dir(dir_path, read_score=False):
    targets = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            src=os.path.join(root, f)
            targets.append(parse_file(src, read_score))
    print("read dir {} done.".format(dir_path))
    return targets

def main():
    parser = argparse.ArgumentParser(description="Result Checker")
    parser.add_argument(
        "--target",
        default="",
        metavar="FILE",
        help="path to ground truth",
        type=str,
    )
    parser.add_argument(
        "--result",
        default="",
        metavar="FILE",
        help="path to detection result",
        type=str,
    )
    args = parser.parse_args()
    gts = parse_dir(args.target)
    dets = parse_dir(args.result, True)
    eval_result = evaluate(dets, gts, FAIR2M_CLASSES)
    print(eval_result)

if __name__ == "__main__":
    main()