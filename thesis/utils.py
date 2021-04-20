import re
import os
from os import path
import time

import numpy as np
import torch
from torchvision import utils as tvutils
from torchvision.transforms import functional as ttf
import matplotlib.pyplot as plt
import matplotlib.collections as pltcollections
import matplotlib.patches as patches
import matplotlib.patheffects as effects
import squarify
from skimage.segmentation import flood
from skimage.filters import sobel

from .models.classification import nearest_neighbors

def recall_tensor(tensor):
    return tensor.detach().cpu().numpy()

def plot_boxes(boxes, color='blue', hl_color = None, hl_width = 5, ax = None):
    if ax is None:
        ax = plt.gca()

    hl_offset = (hl_width - 1) // 2
    highlights = [patches.Rectangle((x, y), w, h) for x, y, w, h in boxes]
    boxes = [patches.Rectangle((x, y), w, h) for x, y, w, h in boxes]

    if hl_color is None: hl_color = 'dark' + color
    highlightcollection = pltcollections.PatchCollection(highlights, facecolor='none', edgecolor=hl_color, linewidth=hl_width)
    boxcollection = pltcollections.PatchCollection(boxes, facecolor='none', edgecolor=color)

    ax.add_collection(highlightcollection)
    ax.add_collection(boxcollection)

def plot_labels(labels, boxes, color='blue', hl_color = None, ax = None):
    if ax is None:
        ax = plt.gca()

    peffects = [effects.withStroke(linewidth=3, foreground=hl_color)] if hl_color is not None else [effects.Normal()]

    for l, b in zip(labels, boxes):
        x, y, _, _ = b
        ax.text(x + 12, y + 12, l, color=color, va='top', ha='left', fontweight='bold', path_effects=peffects)

def build_fig(img, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = [], figsize=(12, 12), ax = None):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    ax.set_axis_off()

    if len(img.shape) == 2:
        img = img[None]
    ax.imshow(img.numpy().transpose((1, 2, 0)), interpolation='nearest')

    plot_boxes(groundtruth, 'lightgreen', 'darkgreen', ax=ax)
    plot_labels(groundtruth_labels, groundtruth, 'lightgreen', 'darkgreen', ax=ax)
    plot_boxes(detections, 'red', ax=ax)
    plot_labels(detection_labels, detections, 'red', ax=ax)

def build_emb_fig(anchors, emb_anchors, positives, emb_positives, figsize=(12, 12)):
    assert len(anchors) <= 10 and len(positives) <= 10, 'Max. 10 supported atm'

    nns = recall_tensor(nearest_neighbors(emb_anchors, emb_positives))

    emb_all = torch.cat((emb_anchors, emb_positives))
    components = recall_tensor(pca(emb_all))

    cmap = plt.cm.tab20(torch.linspace(0, 1, 20))
    cmap[0] = [0.12156863, 0.46666667, 0.70588235, 1.] # for some reason matplotlib gives 0,0,0 instead of this real first element
    c_anchors = cmap[::2][:len(anchors)]
    c_positives = cmap[1::2][:len(positives)]
    c_all = np.concatenate((c_anchors, c_positives))

    fig = plt.figure(figsize=figsize)
    gridsize = max(len(anchors), len(positives))
    extra_rows = 3 if gridsize < 5 else 0
    gs = fig.add_gridspec(gridsize + extra_rows, gridsize)

    anchor_axes = [fig.add_subplot(gs[-1, i]) for i in range(len(anchors))]
    positive_axes = [fig.add_subplot(gs[-3, i]) for i in range(len(positives))]
    scatter_ax = fig.add_subplot(gs[:-3, :])

    scatter_ax.scatter(components[:, 0], components[:, 1], c=c_all)
    scatter_ax.set_xlabel('Principal component 1')
    scatter_ax.set_ylabel('Principal component 2')

    for anch, col, ax in zip(recall_tensor(anchors), c_anchors, anchor_axes):
        ax.imshow(anch.transpose((1, 2, 0)))
        patch = patches.Rectangle((-.5, -.5), anch.shape[-1], anch.shape[-2], color=col, fill=False, lw=6)
        ax.add_patch(patch)
        ax.axis('off')
    
    for pos, col, ax, nearest in zip(recall_tensor(positives), c_positives, positive_axes, nns):
        ax.imshow(pos.transpose((1, 2, 0)))
        patch = patches.Rectangle((-.5, -.5), pos.shape[-1], pos.shape[-2], color=col, fill=False, lw=6)
        ax.add_patch(patch)
        ax.axis('off')

        connection = patches.ConnectionPatch(
            xyA = (pos.shape[-1] // 2, pos.shape[-2]),
            coordsA = ax.transData,
            xyB = (anchors.shape[-1] // 2, 0),
            coordsB = anchor_axes[nearest].transData,
            arrowstyle=patches.ArrowStyle.Curve(), linewidth=2, color=col
        )
        fig.add_artist(connection)
    
    fig.tight_layout(pad=0.5)

def show(img, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = []):
    build_fig(img, detections, groundtruth, detection_labels, groundtruth_labels)
    plt.show()
    plt.close()

def show_multiple(imgs):
    show(tvutils.make_grid(imgs))

def save(img, out, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = []):
    build_fig(img, detections, groundtruth, detection_labels, groundtruth_labels)
    plt.savefig(out, bbox_inches='tight')
    plt.close()

def save_multiple(imgs, out):
    tvutils.save_image(imgs, out)

def save_emb(out, anchors, emb_anchors, positives, emb_positives):
    build_emb_fig(anchors, emb_anchors, positives, emb_positives)
    plt.savefig(out)
    plt.close()

def draw_planogram(boxes, labels, ax = None, xlim = None, ylim = None):
    if ax is None:
        plt.figure(figsize=(12, 9))
        ax = plt.gca()

    if xlim is None:
        xlim = (boxes[:,0].amin(), boxes[:,2].amax())
    if ylim is None:
        ylim = (boxes[:,1].amin(), boxes[:,3].amax())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')

    box_patches = [patches.Rectangle((x1, y1), x2 - x1, y2 - y1) for x1, y1, x2, y2 in boxes]
    box_collection = pltcollections.PatchCollection(box_patches, facecolor='none', edgecolor='black')
    ax.add_collection(box_collection)
    for label, (x1, y1, x2, y2) in zip(labels, boxes):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        ax.text(x, y, label, ha='center', va='center')

def draw_dataset_sample(test_imgs, test_boxes, test_labels, train_imgs, train_labels, figsize = (5, 5)):
    fig = plt.figure(figsize=figsize)
    fig.set_dpi(200)
    gs = fig.add_gridspec(4, 5)
    train_axes = [fig.add_subplot(gs[y, x]) for y in range(4) for x in range(2)]
    test_axes = [fig.add_subplot(gs[:2, 2:]), fig.add_subplot(gs[2:, 2:])]
    for img, label, ax in zip(train_imgs, train_labels, train_axes):
        ax.imshow(img.numpy().transpose((1, 2, 0)), interpolation='bilinear')
        ax.set_title(label)
        ax.axis('off')
    for img, boxes, labels, ax in zip(test_imgs, test_boxes, test_labels, test_axes):
        build_fig(img, groundtruth=boxes, groundtruth_labels=labels, ax=ax)
    plt.show()

def gp_distribution(dataset):
    res = {}
    leaf = {}
    for _, _, hier in dataset:
        for i in range(1, len(hier)+1):
            part = hier[:i]
            key = "/".join(part)
            if key not in res:
                res[key] = 0
                leaf[key] = False
            res[key] += 1
        leaf["/".join(hier)] = True
    return res, leaf

def gp_test_distribution(dataset):
    hierset = []
    for _, anns, _ in dataset:
        for a in anns:
            hier = a.split("/")[:-1]
            hierset.append((None, None, hier))
    return gp_distribution(hierset)

def plot_gp_distribution(dist, leaf):
    sizes = []
    labels = []
    for k, v in leaf.items():
        if v:
            sizes.append(dist[k])
            labels.append(k)
    plt.figure(figsize=(8, 6))
    squarify.plot(sizes=sizes, label=labels)
    plt.show()

def script_dir():
    return path.abspath(path.join(path.dirname(path.realpath(__file__)), '..'))

def rel_path(*parts):
    return path.join(script_dir(), *parts)

def dist_init_file():
    return rel_path('dist_init')

def ensure_dist_file_clean():
    if path.exists(dist_init_file()): # Make sure that the initialization file is clean to avoid unforeseen consequences
        os.remove(dist_init_file())

def trim_module_prefix(state_dict):
    regex = re.compile(r'^module\.(.*)$')
    return {regex.match(k).group(1): v for k, v in state_dict.items()}

def scale_to_tanh(tensor):
    return tensor * 2 - 1

def scale_from_tanh(tensor):
    return (tensor + 1) / 2

def pca(tensors, keepdims = 2):
    u, s, _ = torch.svd(tensors)
    return torch.stack([u[:, i] * s[i] for i in range(keepdims)], dim=1)

def labels_to_tensors(l1, *ln):
    key = list(set(l1).union(*ln))
    conversion = {l: i for i, l in enumerate(key)}
    res = (torch.tensor([conversion[l] for l in lbl], dtype=torch.long) for lbl in [l1, *ln])
    return (*res, key)

def tensors_to_labels(key, *ln):
    res = tuple([key[i] for i in lbl] for lbl in ln)
    return res

def build_mask(img, tolerance=1e-2):
    _, h, w = img.shape
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    gray_image = ttf.rgb_to_grayscale(img).numpy()
    white_corners = [(x, y) for x, y in corners if gray_image[0, y, x] >= 1 - tolerance]
    sobel_image = sobel(gray_image)[0]
    mask = np.full((h, w), False)
    for x, y in white_corners:
        if mask[y, x]: continue
        cfill = flood(sobel_image, (y, x), tolerance=tolerance)
        mask = mask | cfill
    return torch.tensor(mask)

def print_time():
    print(f'-- {time.asctime(time.localtime())} --')
