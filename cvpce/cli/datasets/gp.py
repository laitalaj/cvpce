import random
import re
import os
import shutil

import click
import networkx as nx
import torch
from torchvision import ops as tvops
from matplotlib import pyplot as plt

from ... import datautils, utils, production
from ...defaults import GP_ROOT, GP_TEST_DIR, GP_TEST_VALIDATION_SET, GP_ANN_DIR, GP_BASELINE_ANN_FILE, GP_TRAIN_FOLDERS, GP_PLANO_DIR

@click.group()
def gp():
    '''
    Commands for Grocery Products dataset.

    This command group contains commands for visualizing various aspects of the
    Grocery Products dataset (George et al. 2014)
    and the annotations, planograms and extra data in GP-180 (Tonioni et al. 2017).

    The actual commands under this don't contain help texts,
    sorry about that!
    I'll try to have time to add those in the future.
    '''
    pass

@gp.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=GP_BASELINE_ANN_FILE, show_default=True
)
def visualize_baseline(imgs, annotations):
    data = datautils.GPBaselineDataset(imgs, annotations)
    img, anns = random.choice(data)
    utils.show(img,
        groundtruth=tvops.box_convert(anns['boxes'], 'xyxy', 'xywh')
    )

@gp.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True
)
@click.option('--only', type=str, multiple=True)
def visualize_train(img_dir, only):
    data = datautils.GroceryProductsDataset(img_dir, only=only if len(only) else None, include_annotations=True, include_masks=True)
    img, gen_img, hier, ann = random.choice(data)
    print(' - '.join(hier))
    print(ann)
    mask = utils.scale_from_tanh(gen_img[3])
    utils.show_multiple([utils.scale_from_tanh(img), utils.scale_from_tanh(gen_img[:3]), torch.stack((mask, mask, mask))])

@gp.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True
)
@click.option('--store', type=int)
@click.option('--image', type=int)
def visualize_test(imgs, annotations, store, image):
    dataset = datautils.GroceryProductsTestSet(imgs, annotations)
    if store is None or image is None:
        img, anns, boxes = random.choice(dataset)
    else:
        idx = dataset.get_index_for(store, image)
        if idx is None:
            print(f'No image or annotations for store {store}, image {image}')
            return
        img, anns, boxes = dataset[idx]
    utils.show(img, groundtruth=tvops.box_convert(boxes, 'xyxy', 'xywh'), groundtruth_labels=anns)

@gp.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR, show_default=True
)
def visualize_planoset(img_dir, test_imgs, annotations, planograms):
    data = datautils.PlanogramTestSet(test_imgs, annotations, planograms)
    rebuildset = datautils.GroceryProductsDataset(img_dir, include_annotations=True, resize=False)
    img, anns, boxes, plano = random.choice(data)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8)) if img.shape[2] >= img.shape[1] else plt.subplots(1, 3, figsize=(6, 6))
    fig.set_dpi(300)

    centres = torch.tensor([[(x1 + x2) / 2, -(y1 + y2) / 2] for x1, y1, x2, y2 in plano['boxes']])
    nx.draw(plano['graph'], pos={i: (x.item(), y.item()) for i, (x, y) in enumerate(centres)}, ax=ax1, with_labels=True)
    utils.build_rebuild(plano['boxes'], plano['labels'], rebuildset, ax=ax2)
    utils.build_fig(img, ax=ax3)
    plt.show()

@gp.command()
@click.option(
    '--train-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True
)
def visualize(train_imgs, test_imgs, annotations):
    ann_re = re.compile(r'^(Food/)?(.*?)\..*$')
    def shorten_ann(ann):
        try:
            return ann_re.match(ann).group(2)
        except AttributeError:
            print(f'Malformed annotation: {ann}')
            return ann

    train_set = datautils.GroceryProductsDataset(train_imgs, include_annotations=True, random_crop=False, resize=False)
    test_set = datautils.GroceryProductsTestSet(test_imgs, annotations)
    test_imgs, test_anns, test_boxes = zip(*[random.choice(test_set) for _ in range(2)])
    test_boxes = [tvops.box_convert(boxes, 'xyxy', 'xywh') for boxes in test_boxes]

    uniq_anns = set(test_anns[0]) | set(test_anns[1])
    train_imgs = []
    train_anns = []
    for ann in uniq_anns:
        idx = train_set.index_for_ann(ann)
        if idx is None: continue
        img, _, _, ann = train_set[idx]
        train_imgs.append(img)
        train_anns.append(ann)
    if len(train_imgs) < 8:
        more_imgs, _, _, more_anns = zip(*[random.choice(train_set) for _ in range(8 - len(train_imgs))])
        train_imgs += more_imgs
        train_anns += more_anns

    test_anns = [[shorten_ann(ann) for ann in anns] for anns in test_anns]
    train_anns = [shorten_ann(ann) for ann in train_anns]

    utils.draw_dataset_sample(test_imgs, test_boxes, test_anns, train_imgs, train_anns)

@gp.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True
)
@click.option('--only', type=str, multiple=True)
def train_distribution(img_dir, only):
    if not len(only): only = None
    data = datautils.GroceryProductsDataset(img_dir, only=only, random_crop=False)
    dist, leaf = utils.gp_distribution(data)
    for h, c in dist.items():
        print(f'{h}: {c} ({leaf[h]})')

@gp.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True
)
@click.option('--only', type=click.Choice(('none', 'test', 'val', 'keep2', 'skip2')), default='none', show_default=True)
def test_distribution(imgs, annotations, only):
    only_list = None
    skip_list = None
    if only == 'test':
        skip_list = GP_TEST_VALIDATION_SET
    elif only == 'val':
        only_list = GP_TEST_VALIDATION_SET
    elif only == 'keep2':
        only_list = 2
    elif only == 'skip2':
        skip_list = 2

    data = datautils.GroceryProductsTestSet(imgs, annotations, only=only_list, skip=skip_list)
    dist, leaf = utils.gp_test_distribution(data)
    for h, c in dist.items():
        print(f'{h}: {c} ({leaf[h]})')
    utils.plot_gp_distribution(dist, leaf)

@gp.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=utils.rel_path(*GP_ROOT, 'Training', 'Food'), show_default=True
)
@click.option(
    '--out-dir', type=click.Path(exists=False), default=utils.rel_path(*GP_ROOT, 'Training', 'Food_Fixed'), show_default=True
)
@click.option('--dry-run/--no-dry-run', default=False, show_default=True)
def fix(source_dir, out_dir, dry_run):
    renamed_re = re.compile(r'food_(\d+).jpg')
    to_search = [source_dir]
    hierarchies = [[]]
    print('Fixing GP...')
    while len(to_search):
        current_path = to_search.pop()
        current_hierarchy = hierarchies.pop()
        print(f'{current_path}...')

        files = []
        for entry in os.scandir(current_path):
            if entry.is_dir(follow_symlinks=False): # not following symlinks here to avoid possibily infinite looping
                to_search.append(entry.path)
                hierarchies.append(current_hierarchy + [entry.name])
            elif entry.is_file():
                match = renamed_re.match(entry.name)
                if match is None: continue
                files.append((int(match.group(1)), entry))

        if not files: continue

        _, files = zip(*sorted(files))
        new_names = sorted([f'{i}.jpg' for i in range(1, len(files))]) # the original annotations have JPGs and jpegs, but Tonioni's use only jpgs

        out_path = os.path.join(out_dir, *current_hierarchy)
        if dry_run:
            i = 0
        else:
            os.makedirs(out_path)
        print(f'{"(Not) " if dry_run else ""}Copying {len(files) - 1} files to {out_path}...')
        for f, new in zip(files[1:], new_names): # the first entry is always garbage
            if dry_run:
                if i == 0:
                    print(f'{f.path} -> {os.path.join(out_path, new)}')
                    i = 25
                else:
                    i -= 1
            else:
                shutil.copy(f.path, os.path.join(out_path, new))
    print('Done!')

@gp.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR, show_default=True
)
def planogram_test(imgs, annotations, planograms):
    data = datautils.PlanogramTestSet(imgs, annotations, planograms, only=GP_TEST_VALIDATION_SET)
    img, anns, boxes, plano = random.choice(data)
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    utils.draw_planogram(plano['boxes'], plano['labels'], ax=ax1)
    centres = torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in plano['boxes']])
    nx.draw(plano['graph'], pos={i: (x.item(), y.item()) for i, (x, y) in enumerate(centres)}, ax=ax2, with_labels=True)
    #utils.build_fig(img, groundtruth=boxes, groundtruth_labels=anns, ax=ax3)
    utils.build_fig(img, ax=ax3) # TODO: Jostain syystä ei toimi groundtruthien piirto täs .__.
    plt.show()

    comparator = production.PlanogramComparator()
    res = comparator.compare(plano, {'boxes': boxes, 'labels': anns})
    print(res)

@gp.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True
)
@click.option('--only', type=str, multiple=True)
def mask_test(img_dir, only):
    data = datautils.GroceryProductsDataset(img_dir, only=only if len(only) else None)
    img, _, _ = random.choice(data)
    img = utils.scale_from_tanh(img)
    mask = utils.build_mask(img)
    utils.show_multiple([img, torch.stack((mask, mask, mask))])
