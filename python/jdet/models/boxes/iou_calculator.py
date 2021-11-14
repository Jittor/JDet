import jittor as jt
from jdet.ops import box_iou_rotated, box_iou_rotated_v1
from jdet.utils.registry import BOXES
import numpy as np

@BOXES.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False, version=0):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned, version=version)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

@BOXES.register_module()
class BboxOverlaps2D_v1:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False, version=1):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, shape (m, 5) in <x1, y1, x2, y2, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned, version=version)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


@BOXES.register_module()
class BboxOverlaps2D_rotated:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]

        assert mode == "iou" and is_aligned == False
        # TODO: add giou....
        return bbox_overlaps_rotated(bboxes1, bboxes2)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

@BOXES.register_module()
class BboxOverlaps2D_rotated_v1:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format.
            bboxes2 (Tensor): bboxes have shape (m, 5) in <x, y, w, h, a>
                format, or shape (m, 5) in <x, y, w, h, a, score> format, or be
                empty. If ``is_aligned `` is ``True``, then m and n must be
                equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        assert bboxes1.size(-1) in [0, 5, 6]
        assert bboxes2.size(-1) in [0, 5, 6]
        if bboxes2.size(-1) == 6:
            bboxes2 = bboxes2[..., :5]
        if bboxes1.size(-1) == 6:
            bboxes1 = bboxes1[..., :5]

        assert mode == "iou" and is_aligned == False
        # TODO: add giou....
        return bbox_overlaps_rotated(bboxes1, bboxes2, version=1)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str

def bbox_overlaps_rotated(rboxes1, rboxes2, version=0):
    if version == 0:
        ious = box_iou_rotated(rboxes1.float(), rboxes2.float())
    else:
        ious = box_iou_rotated_v1(rboxes1.float(), rboxes2.float())
    return ious

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6, version=0):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = jt.float([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = jt.float([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = jt.zeros((0, 4))
        >>> nonempty = jt.float([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return jt.zeros(batch_shape + (rows,), bboxes1.dtype)
        else:
            return jt.zeros(batch_shape + (rows, cols), bboxes1.dtype)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0] + version) * (
        bboxes1[..., 3] - bboxes1[..., 1] + version)
    area2 = (bboxes2[..., 2] - bboxes2[..., 0] + version) * (
        bboxes2[..., 3] - bboxes2[..., 1] + version)

    if is_aligned:
        lt = jt.maximum(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = jt.minimum(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt + version).clamp(min_v=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = jt.maximum(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = jt.maximum(bboxes1[..., :, :2].unsqueeze(-2),
                        bboxes2[..., :, :2].unsqueeze(-3))  # [B, rows, cols, 2]
        rb = jt.minimum(bboxes1[..., :, 2:].unsqueeze(-2),
                        bboxes2[..., :, 2:].unsqueeze(-3))  # [B, rows, cols, 2]

        wh = (rb - lt + version).clamp(min_v=0)  # [B, rows, cols, 2]
        
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1.unsqueeze(-1) + area2.unsqueeze(-2) - overlap
        else:
            union = area1.unsqueeze(-1)
        if mode == 'giou':
            enclosed_lt = jt.minimum(bboxes1[..., :, :2].unsqueeze(-2),
                                     bboxes2[..., :, :2].unsqueeze(-3))
            enclosed_rb = jt.maximum(bboxes1[..., :, 2:].unsqueeze(-2),
                                     bboxes2[..., :, 2:].unsqueeze(-3))

    union = jt.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min_v=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = jt.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def bbox_overlaps_np(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(
            y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious
