import random

import click
from torchvision import ops as tvops

from ... import datautils, utils
from ...defaults import GROZI_ROOT

@click.group()
def grozi():
    pass

@grozi.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def visualize_train(root):
    dataset = datautils.GroZiDataset(root)
    img, _ = random.choice(dataset)
    utils.show(img)

@grozi.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
@click.option('--select-from', type=click.Choice(['none', 'min', 'max']), default='none')
def visualize_test(root, select_from):
    dataset = datautils.GroZiTestSet(root)
    if select_from == 'min':
        idxset = dataset.least_annotated()
        print(f'There are {len(idxset)} least-annotated images')
    elif select_from == 'max':
        idxset = dataset.most_annotated()
        print(f'There are {len(idxset)} most-annotated images')
    else:
        idxset = range(len(dataset))
        print(f'There are {len(dataset)} images')
    img, anns, boxes = dataset[random.choice(idxset)]
    print(f'Annotations in image: {len(anns)}')
    utils.show(img, groundtruth=tvops.box_convert(boxes, 'xyxy', 'xywh'), groundtruth_labels=anns)

@grozi.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def visualize(root):
    train_set = datautils.GroZiDataset(root)
    test_set = datautils.GroZiTestSet(root)
    test_imgs, test_anns, test_boxes = zip(*[random.choice(test_set) for _ in range(2)])
    test_boxes = [tvops.box_convert(boxes, 'xyxy', 'xywh') for boxes in test_boxes]
    test_anns = [[ann.item() for ann in anns] for anns in test_anns]

    uniq_anns = set(test_anns[0]) | set(test_anns[1])
    train_imgs = []
    train_anns = []
    for ann in uniq_anns:
        idx = train_set.index_for_ann(ann)
        img, ann = train_set[idx]
        train_imgs.append(img)
        train_anns.append(ann)
    if len(train_imgs) < 8:
        more_imgs, more_anns = zip(*[random.choice(train_set) for _ in range(8 - len(train_imgs))])
        train_imgs += more_imgs
        train_anns += more_anns
    utils.draw_dataset_sample(test_imgs, test_boxes, test_anns, train_imgs, train_anns)

@grozi.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def extract_test_images(root):
    datautils.extract_grozi_test_imgs(root)
