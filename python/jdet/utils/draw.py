# from OBBDetection
import numpy as np
from numpy import pi
import time 
import cv2 

from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import os.path as osp
import matplotlib.colors as mpl_colors
import matplotlib
matplotlib.use("Agg")

from collections.abc import Iterable


def draw_hbb(ax,
             bboxes,
             texts,
             color,
             thickness=1.,
             font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    patches, edge_colors = [], []
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        if texts is not None:
            ax.text(xmin,
                    ymin,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    fontsize=font_size,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)


def draw_obb(ax,
             bboxes,
             texts,
             color,
             thickness=1.,
             font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    ctr, w, h, t = np.split(bboxes, (2, 3, 4), axis=1)
    Cos, Sin = np.cos(t), np.sin(t)
    vec1 = np.concatenate(
        [-w/2 * Cos, w/2 * Sin], axis=1)
    vec2 = np.concatenate(
        [-h/2 * Sin, -h/2 * Cos], axis=1)
    anchors = ctr + vec1 + vec2
    angles = -t * 180 / pi
    new_obbs = np.concatenate([anchors, w, h, angles], axis=1)

    patches, edge_colors = [], []
    for i, bbox in enumerate(new_obbs):
        x, y, w, h, angle = bbox
        if texts is not None:
            ax.text(x,
                    y,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    rotation=angle,
                    rotation_mode='anchor',
                    fontsize=font_size,
                    transform_rotates_text=True,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Rectangle((x, y), w, h, angle))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)


def draw_poly(ax,
              bboxes,
              texts,
              color,
              thickness=1.,
              font_size=10):
    if texts is not None:
        assert len(texts) == len(bboxes)

    pts = bboxes.reshape(-1, 4, 2)
    top_pts_idx = np.argsort(pts[..., 1], axis=1)[:, :2]
    top_pts_idx = top_pts_idx[..., None].repeat(2, axis=2)
    top_pts = np.take_along_axis(pts, top_pts_idx, axis=1)

    x_sort_idx = np.argsort(top_pts[..., 0], axis=1)
    left_idx, right_idx = x_sort_idx[:, :1], x_sort_idx[:, 1:]
    left_idx = left_idx[..., None].repeat(2, axis=2)
    left_pts = np.take_along_axis(top_pts, left_idx, axis=1).squeeze(1)
    right_idx = right_idx[..., None].repeat(2, axis=2)
    right_pts = np.take_along_axis(top_pts, right_idx, axis=1).squeeze(1)

    x2 = right_pts[:, 1] - left_pts[:, 1]
    x1 = right_pts[:, 0] - left_pts[:, 0]
    angles = np.arctan2(x2, x1) / pi * 180

    patches, edge_colors = [], []
    for i, (pt, anchor, angle) in enumerate(zip(
        pts, left_pts, angles)):
        x, y = anchor
        if texts is not None:
            ax.text(x,
                    y,
                    texts[i],
                    bbox={
                        'alpha': 0.5,
                        'pad': 0.7,
                        'facecolor': color,
                        'edgecolor': 'none'
                    },
                    color='white',
                    rotation=angle,
                    rotation_mode='anchor',
                    fontsize=font_size,
                    transform_rotates_text=True,
                    verticalalignment='bottom',
                    horizontalalignment='left')

        patches.append(Polygon(pt))
        edge_colors.append(color)

    if patches:
        p = PatchCollection(
            patches,
            facecolors='none',
            edgecolors=edge_colors,
            linewidths=thickness)
        ax.add_collection(p)


def single_color_val(color):
    '''Convert single color to rgba format defined in matplotlib.
    A single color can be Iterable, int, float and str. All int
    will be divided by 255 to follow the color defination in
    matplotlib.
    '''
    # Convert Iterable, int, float to list.
    if isinstance(color, str):
        color = color.split('$')[0]
    elif isinstance(color, Iterable):
        color = [c/255 if isinstance(c, int) else c for c in color]
    elif isinstance(color, int):
        color = (color/255, color/255, color/255)
    elif isinstance(color, float):
        color = (color, color, color)

    # Assert wheather color is valid.
    assert mpl_colors.is_color_like(color) , \
            f'{color} is not a legal color in matplotlib.colors'
    return mpl_colors.to_rgb(color)


def colors_val(colors):
    '''Convert colors to rgba format. Colors should be Iterable or str.
    If colors is str, functions will first try to treat colors as a file
    and read lines from it. If the file is not existing, the function
    will split the str by '|'.
    '''
    if isinstance(colors, str):
        if osp.isfile(colors):
            with open(colors, 'r') as f:
                colors = [line.strip() for line in f]
        else:
            colors = colors.split('|')
    return [single_color_val(c) for c in colors]


def random_colors(num, cmap=None):
    '''Random generate colors.

    Args:
        num (int): number of colors to generate.
        cmap (matplotlib cmap): refer to matplotlib cmap.

    Returns:
        several colors.
    '''
    if cmap is None:
        return colors_val(np.random.rand(num, 3))
    else:
        return colors_val(cmap(np.random.rand(num)))

def plt_init(width, height):
    EPS = 1e-2
    win_name = str(time.time())
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    return ax, fig

def get_img_from_fig(fig, width, height):
    stream, _ = fig.canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype=np.uint8)
    img_rgba = buffer.reshape(height, width, 4)
    img, _ = np.split(img_rgba, [3], axis=2)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def draw_bboxes(img,
                  bboxes,
                  labels=None,
                  scores=None,
                  class_names=None,
                  score_thr=0,
                  colors='green',
                  thickness=1,
                  with_text=True,
                  font_size=10,
                  out_file=None):

    if isinstance(img, np.ndarray):
        img = np.ascontiguousarray(img)
    else:
        img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert isinstance(img,np.ndarray), "image must be a numpy array!"
    assert isinstance(bboxes,np.ndarray), "boxes must be a numpy array!"
    assert labels is None or (labels.shape[0]==bboxes.shape[0] and labels.ndim==1)
    assert scores is None or (scores.shape[0]==bboxes.shape[0] and scores.ndim==1)
    assert bboxes.shape[1] in [4,5,8] and bboxes.ndim==2

    draw_funcs = {
        4 : draw_hbb,
        5 : draw_obb,
        8 : draw_poly
    }
    draw_func = draw_funcs[bboxes.shape[1]]

    if scores is None:
        with_score = False
    else:
        bboxes = np.concatenate([bboxes, scores[:, None]], axis=1)
        with_score = True

    if labels is None:
        bboxes = [bboxes]
    else:
        bboxes = [bboxes[labels == i] for i in range(labels.max()+1)]

    if colors == 'random':
        colors = random_colors(len(bboxes))
    else:
        colors = colors_val(colors)
        if len(colors) == 1:
            colors = colors * len(bboxes)
        assert len(colors) >= len(bboxes)


    height, width = img.shape[:2]
    ax, fig = plt_init(width, height)
    plt.imshow(img)

    for i, cls_bboxes in enumerate(bboxes):
        if with_score:
            cls_bboxes = cls_bboxes[cls_bboxes[:, -1] > score_thr]
            cls_bboxes, cls_scores = cls_bboxes[:, :-1], cls_bboxes[:, -1]

        if not with_text:
            texts = None
        else:
            texts = []
            for j in range(len(cls_bboxes)):
                text = f'cls: {i}' if class_names is None else class_names[i]
                if with_score:
                    text += f'|{cls_scores[j]:.02f}'
                texts.append(text)

        draw_func(ax, cls_bboxes, texts, colors[i], thickness, font_size)

    drawed_img = get_img_from_fig(fig, width, height)
    

    if out_file is not None:
        cv2.imwrite(out_file, drawed_img)

    plt.close(fig)
    return drawed_img
