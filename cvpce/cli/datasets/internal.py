import random
import os

import click
import cv2
import torch
from matplotlib import pyplot as plt

from ... import datautils, utils

@click.group()
def internal():
    pass

@internal.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
def visualize_train(img_dir):
    data = datautils.InternalTrainSet(img_dir, include_annotations=True, include_masks=True)
    img, gen_img, hier, ann = random.choice(data)
    print(' - '.join(hier))
    print(ann)
    mask = utils.scale_from_tanh(gen_img[3])
    utils.show_multiple([utils.scale_from_tanh(img), utils.scale_from_tanh(gen_img[:3]), torch.stack((mask, mask, mask))])

@internal.command()
@click.option('--dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def visualize_planoset(dir):
    data = datautils.InternalPlanoSet(dir)
    img, plano = random.choice(data)

    __, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    reduced_labels = list(set(plano['labels']))
    utils.draw_planogram(plano['boxes'], [reduced_labels.index(l) for l in plano['labels']], ax=ax1)
    utils.build_fig(img, ax=ax2)
    plt.show()

@internal.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def visualize(root):
    test_set = datautils.InternalPlanoSet(root)
    train_set = datautils.InternalTrainSet(os.path.join(root, 'ConvertedProducts'), include_annotations=True, random_crop=False, resize=False)

    test_imgs = [random.choice(test_set)[0] for _ in range(2)]
    train_imgs, _, _, train_anns = zip(*[random.choice(train_set) for _ in range(8)])
    train_anns = [ann[:8] for ann in train_anns]

    print(f'Different products: {len(set(train_set.annotations))}')
    utils.draw_dataset_sample(test_imgs, [[],[]], [[],[]], train_imgs, train_anns)

@internal.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
def iter_train(img_dir):
    data = datautils.InternalTrainSet(img_dir, include_annotations=True, include_masks=True)
    for i in range(len(data)):
        if i % 100 == 0:
            print(i)
        try:
            data[i]
        except cv2.error:
            continue
