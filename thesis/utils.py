import re
import os
from os import path
import time

import numpy as np
import torch
from torchvision import utils as tvutils
import matplotlib.pyplot as plt
import matplotlib.collections as pltcollections
import matplotlib.patches as patches

from .models.classification import nearest_neighbors

def recall_tensor(tensor):
    return tensor.detach().cpu().numpy()

def plot_boxes(boxes, color='blue', hl_color = None, hl_width = 5):
    hl_offset = (hl_width - 1) // 2
    highlights = [patches.Rectangle((x, y), w, h) for x, y, w, h in boxes]
    boxes = [patches.Rectangle((x, y), w, h) for x, y, w, h in boxes]

    if hl_color is None: hl_color = 'dark' + color
    highlightcollection = pltcollections.PatchCollection(highlights, facecolor='none', edgecolor=hl_color, linewidth=hl_width)
    boxcollection = pltcollections.PatchCollection(boxes, facecolor='none', edgecolor=color)

    plt.gca().add_collection(highlightcollection)
    plt.gca().add_collection(boxcollection)

def plot_labels(labels, boxes, color='blue'):
    for l, b in zip(labels, boxes):
        x, y, _, _ = b
        plt.text(x + 1, y - 1, l, color=color, va='top', ha='left')

def build_fig(img, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = [], figsize=(12, 12)):
    plt.figure(figsize=figsize)
    plt.axis('off')

    if len(img.shape) == 2:
        img = img[None]
    plt.imshow(img.numpy().transpose((1, 2, 0)), interpolation='nearest')

    plot_boxes(groundtruth, 'green')
    plot_labels(groundtruth_labels, groundtruth, 'green')
    plot_boxes(detections, 'red')
    plot_labels(detection_labels, detections, 'red')

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
    
    for pos, col, ax, nearest in zip(scale_from_tanh(recall_tensor(positives)), c_positives, positive_axes, nns):
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

def print_time():
    print(f'-- {time.asctime(time.localtime())} --')
