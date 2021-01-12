from os import path

import matplotlib.pyplot as plt
import matplotlib.collections as pltcollections
import matplotlib.patches as patches

def recall_tensor(tensor):
    return tensor.detach().cpu().numpy()

def plot_boxes(boxes, color='blue'):
    boxes = [patches.Rectangle((x, y), w, h) for x, y, w, h in boxes]
    boxcollection = pltcollections.PatchCollection(boxes, facecolor='none', edgecolor=color)
    plt.gca().add_collection(boxcollection)

def plot_labels(labels, boxes, color='blue'):
    for l, b in zip(labels, boxes):
        x, y, _, _ = b
        plt.text(x + 1, y - 1, l, color=color, va='top', ha='left')

def build_fig(img, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = []):
    plt.figure()

    if len(img.shape) == 2:
        img = img[None]
    plt.imshow(img.numpy().transpose((1, 2, 0)), interpolation='nearest')

    plot_boxes(groundtruth, 'green')
    plot_labels(groundtruth_labels, groundtruth, 'green')
    plot_boxes(detections, 'red')
    plot_labels(detection_labels, detections, 'red')

def show(img, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = []):
    build_fig(img, detections, groundtruth, detection_labels, groundtruth_labels)
    plt.show()
    plt.close()

def save(img, out, detections = [], groundtruth = [], detection_labels = [], groundtruth_labels = []):
    build_fig(img, detections, groundtruth, detection_labels, groundtruth_labels)
    plt.savefig(out)
    plt.close()

def script_dir():
    return path.abspath(path.dirname(path.realpath(__file__)))

def rel_path(*parts):
    return path.join(script_dir(), *parts)
