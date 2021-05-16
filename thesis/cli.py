import os
import random
import re
import shutil
from functools import partial

import click
import cv2
import torch
import torch.multiprocessing as mp
import pycocotools.cocoeval as cocoeval
import torchvision.models as tmodels
import torchvision.datasets as dsets
import torchvision.ops as tvops
import torchvision.transforms as tforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from . import datautils
from . import utils
from . import proposals_training, proposals_eval
from . import classification_training, classification_eval
from . import detection_eval
from . import planograms
from . import production
from . import hyperopt
from .models import proposals, classification

DATA_DIR = ('..', 'data')

COCO_IMG_DIR = utils.rel_path(*DATA_DIR, 'coco', 'val2017')
COCO_ANNOTATION_FILE = utils.rel_path(*DATA_DIR, 'coco', 'annotations', 'instances_val2017.json')

SKU110K_IMG_DIR = utils.rel_path(*DATA_DIR, 'SKU110K_fixed', 'images')
SKU110K_ANNOTATION_FILE = utils.rel_path(*DATA_DIR, 'SKU110K_fixed', 'annotations', 'annotations_val.csv')
SKU110K_SKIP = [
    'test_274.jpg', 'train_882.jpg', 'train_924.jpg', 'train_4222.jpg', 'train_5822.jpg', # corrupted images, won't load, TODO: re-export test for fairness
    'train_789.jpg', 'train_5007.jpg', 'train_6090.jpg', 'train_7576.jpg', # corrupted images, will load
    'train_104.jpg', 'train_890.jpg', 'train_1296.jpg', 'train_3029.jpg', 'train_3530.jpg', 'train_3622.jpg', 'train_4899.jpg', 'train_6216.jpg', 'train_7880.jpg', # missing most ground truth boxes
    'train_701.jpg', 'train_6566.jpg', # very poor images
]


GP_ROOT = (*DATA_DIR, 'Grocery_products')
GP_TRAIN_FOLDERS = (utils.rel_path(*GP_ROOT, 'Training'),)
# GP_TRAIN_FOLDERS = (utils.rel_path(*GP_ROOT),) # For reading index from file
''' Ye olde folders, before I found Tonioni's fixed GP set
GP_TRAIN_FOLDERS = (
    utils.rel_path(*GP_ROOT, 'Training'),
    utils.rel_path(*DATA_DIR, 'Planogram_Dataset', 'extra_products'),
)
'''
GP_TEST_DIR = utils.rel_path(*GP_ROOT, 'Testing')
GP_ANN_DIR = utils.rel_path(*DATA_DIR, 'Planogram_Dataset', 'annotations')
GP_BASELINE_ANN_FILE = utils.rel_path(*DATA_DIR, 'Baseline', 'Grocery_products_coco_gt_object.csv')
GP_PLANO_DIR = utils.rel_path(*DATA_DIR, 'Planogram_Dataset', 'planograms')
GP_TEST_VALIDATION_SET = ['s1_15.csv', 's2_3.csv', 's2_30.csv', 's2_143.csv', 's2_157.csv', 's3_111.csv', 's3_260.csv', 's5_55.csv']
GP_TEST_VALIDATION_SET_SIZE = 2
GP_PLANO_VALIDATION_SET = [f'{s.split(".")[0]}.json' for s in GP_TEST_VALIDATION_SET]

GROZI_ROOT = utils.rel_path(*DATA_DIR, 'GroZi-120')

MODEL_DIR = ('..', 'models')
PRETRAINED_GAN_FILE = utils.rel_path(*MODEL_DIR, 'pretrained_dihe_gan.tar')
ENCODER_FILE = utils.rel_path(*MODEL_DIR, 'encoder.tar')

OUT_DIR = utils.rel_path('out')

@click.group()
def cli():
    pass

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=COCO_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=COCO_ANNOTATION_FILE
)
def visualize_coco(imgs, annotations):
    data = dsets.CocoDetection(root=imgs, annFile=annotations, transform=tforms.ToTensor())
    img, anns = random.choice(data)
    utils.show(img, groundtruth=[ann['bbox'] for ann in anns])

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option('--index')
@click.option(
    '--method',
    type=click.Choice(['normal', 'kant', 'simple']),
    default='simple'
)
@click.option('--flip/--no-flip', default=False)
@click.option('--gaussians/--no-gaussians', default=True)
@click.option('--model', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('--conf-thresh', type=float, default=0.45)
@click.option('--save', type=click.Path(writable=True))
@click.option('--save-gaussians', type=click.Path(writable=True))
def visualize_sku110k(imgs, annotations, index, method, flip, gaussians, model, conf_thresh, save, save_gaussians):
    gauss_methods = {
        'normal': {'gauss_generate_method': datautils.generate_via_multivariate_normal, 'gauss_join_method': datautils.join_via_max},
        'kant': {'gauss_generate_method': datautils.generate_via_kant_method, 'gauss_join_method': datautils.join_via_replacement},
        'simple': {'gauss_generate_method': datautils.generate_via_simple_and_scaled, 'gauss_join_method': datautils.join_via_max, 'tanh': True},
    }
    data = datautils.SKU110KDataset(imgs, annotations, flip_chance=0, **gauss_methods[method])
    if index is None:
        img, anns = random.choice(data)
    else:
        digit_re = re.compile(r'^\d+$')
        if digit_re.match(index) is not None:
            index = int(index)
        else:
            index = data.index_for_name(index)
        img, anns = data[index]
    if flip:
        img, anns = datautils.sku110k_flip(img, anns)
    prediction = []
    if model is not None:
        model = proposals_eval.load_gln(model, False)
        model_result = model(img[None].cuda())
        prediction = utils.recall_tensor(model_result[0]['boxes'][model_result[0]['scores'] > conf_thresh])
    utils.show(img,
        detections=[[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in prediction],
        groundtruth=[] if model is not None else [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in anns['boxes']]
    ) # TODO: Torch has a built in function for converting between coordinate systems
    if save is not None:
        utils.save(img, save, groundtruth=[[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in anns['boxes']])
    if gaussians:
        print(f'Gaussians: Min {anns["gaussians"].amin()}, Max {anns["gaussians"].amax()}')
        utils.show(anns['gaussians'])
    if save_gaussians is not None:
        utils.save(anns['gaussians'], save_gaussians)

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=GP_BASELINE_ANN_FILE
)
def visualize_gp_baseline(imgs, annotations):
    data = datautils.GPBaselineDataset(imgs, annotations)
    img, anns = random.choice(data)
    utils.show(img,
        groundtruth=tvops.box_convert(anns['boxes'], 'xyxy', 'xywh')
    )

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option('--index', type=int)
def visualize_discriminator_target(imgs, annotations, index):
    data = datautils.TargetDomainDataset(imgs, annotations)
    print(f'Dataset size: {len(data)} -- Bbox index: {data.bbox_index[:3]} ... {data.bbox_index[-3:]}')
    img = random.choice(data) if index is None else data[index]
    utils.show(img)

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--only', type=str, multiple=True)
def visualize_gp(img_dir, only):
    data = datautils.GroceryProductsDataset(img_dir, only=only if len(only) else None, include_annotations=True, include_masks=True)
    img, gen_img, hier, ann = random.choice(data)
    print(' - '.join(hier))
    print(ann)
    mask = utils.scale_from_tanh(gen_img[3])
    utils.show_multiple([utils.scale_from_tanh(img), utils.scale_from_tanh(gen_img[:3]), torch.stack((mask, mask, mask))])

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--store', type=int)
@click.option('--image', type=int)
def visualize_gp_test(imgs, annotations, store, image):
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

@cli.command()
@click.option(
    '--train-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
def visualize_gp_set(train_imgs, test_imgs, annotations):
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

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--only', type=str, multiple=True)
def gp_distribution(img_dir, only):
    if not len(only): only = None
    data = datautils.GroceryProductsDataset(img_dir, only=only, random_crop=False)
    dist, leaf = utils.gp_distribution(data)
    for h, c in dist.items():
        print(f'{h}: {c} ({leaf[h]})')

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--only', type=click.Choice(('none', 'test', 'val', 'keep2', 'skip2')), default='none')
def gp_test_distribution(imgs, annotations, only):
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

@cli.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=utils.rel_path(*GP_ROOT, 'Training', 'Food')
)
@click.option(
    '--out-dir', type=click.Path(exists=False), default=utils.rel_path(*GP_ROOT, 'Training', 'Food_Fixed')
)
@click.option('--dry-run/--no-dry-run', default=False)
def fix_gp(source_dir, out_dir, dry_run):
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

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
def visualize_internal_train(img_dir):
    data = datautils.InternalTrainSet(img_dir, include_annotations=True, include_masks=True)
    img, gen_img, hier, ann = random.choice(data)
    print(' - '.join(hier))
    print(ann)
    mask = utils.scale_from_tanh(gen_img[3])
    utils.show_multiple([utils.scale_from_tanh(img), utils.scale_from_tanh(gen_img[:3]), torch.stack((mask, mask, mask))])

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
def iter_internal_train(img_dir):
    data = datautils.InternalTrainSet(img_dir, include_annotations=True, include_masks=True)
    for i in range(len(data)):
        if i % 100 == 0:
            print(i)
        try:
            data[i]
        except cv2.error:
            continue

@cli.command()
@click.option('--dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def visualize_internal_planoset(dir):
    data = datautils.InternalPlanoSet(dir)
    img, plano = random.choice(data)

    __, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    reduced_labels = list(set(plano['labels']))
    utils.draw_planogram(plano['boxes'], [reduced_labels.index(l) for l in plano['labels']], ax=ax1)
    utils.build_fig(img, ax=ax2)
    plt.show()

@cli.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
def visualize_internal_set(root):
    test_set = datautils.InternalPlanoSet(root)
    train_set = datautils.InternalTrainSet(os.path.join(root, 'ConvertedProducts'), include_annotations=True, random_crop=False, resize=False)

    test_imgs = [random.choice(test_set)[0] for _ in range(2)]
    train_imgs, _, _, train_anns = zip(*[random.choice(train_set) for _ in range(8)])
    train_anns = [ann[:8] for ann in train_anns]

    print(f'Different products: {len(set(train_set.annotations))}')
    utils.draw_dataset_sample(test_imgs, [[],[]], [[],[]], train_imgs, train_anns)

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR
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

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR
)
def visualize_gp_planoset(img_dir, test_imgs, annotations, planograms):
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

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--only', type=str, multiple=True)
def gp_mask_test(img_dir, only):
    data = datautils.GroceryProductsDataset(img_dir, only=only if len(only) else None)
    img, _, _ = random.choice(data)
    img = utils.scale_from_tanh(img)
    mask = utils.build_mask(img)
    utils.show_multiple([img, torch.stack((mask, mask, mask))])

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
def iter_sku110k(imgs, annotations):
    data = datautils.SKU110KDataset(imgs, annotations, include_gaussians=False)
    loader = DataLoader(data, batch_size=1, num_workers=8, collate_fn=datautils.sku110k_collate_fn)
    for i, d in enumerate(loader):
        if i % 100 == 0:
            print(i)

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=COCO_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=COCO_ANNOTATION_FILE
)
@click.option(
    '--method',
    type=click.Choice(['normal', 'kant']),
    default='normal'
)
def visualize_gaussians(imgs, annotations, method):
    gauss_methods = {
        'normal': {'generate_method': datautils.generate_via_multivariate_normal(), 'join_method': datautils.join_via_max},
        'kant': {'generate_method': datautils.generate_via_kant_method(), 'join_method': datautils.join_via_replacement},
    }
    data = dsets.CocoDetection(root=imgs, annFile=annotations, transform=tforms.ToTensor())
    img, anns = random.choice(data)
    utils.show(img, groundtruth=[ann['bbox'] for ann in anns])

    _, h, w = img.shape
    coco_to_retina = lambda bbox: torch.tensor([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
    gauss = datautils.generate_gaussians(w, h, [coco_to_retina(ann['bbox']) for ann in anns], **gauss_methods[method])
    utils.show(gauss)

@cli.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def extract_grozi_test_images(root):
    datautils.extract_grozi_test_imgs(root)

@cli.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def visualize_grozi_train(root):
    dataset = datautils.GroZiDataset(root)
    img, _ = random.choice(dataset)
    utils.show(img)

@cli.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
@click.option('--select-from', type=click.Choice(['none', 'min', 'max']), default='none')
def visualize_grozi(root, select_from):
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

@cli.command()
@click.option('--root', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True), default=GROZI_ROOT)
def visualize_grozi_set(root):
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

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=COCO_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=COCO_ANNOTATION_FILE
)
@click.option(
    '--score-threshold',
    type=float,
    default=0.05
)
@click.option(
    '--limit',
    type=int,
    default=-1
)
def retinanet_coco_test(imgs, annotations, score_threshold, limit):
    model = tmodels.detection.retinanet_resnet50_fpn(pretrained=True).cuda()
    model.eval()

    data = dsets.CocoDetection(root=imgs, annFile=annotations, transform=tforms.ToTensor())

    evaluated = []
    results = []
    for i, datum in enumerate(data):
        if i % 100 == 0:
            print(i)
        if limit >= 0 and i >= limit:
            break

        img, anns = datum
        predictions = model(img.unsqueeze(0).cuda())[0]

        keep = predictions['scores'] > score_threshold
        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in utils.recall_tensor(predictions['boxes'][keep])]
        scores = utils.recall_tensor(predictions['scores'][keep])
        labels = utils.recall_tensor(predictions['labels'][keep])

        img_id = data.ids[i]
        results += [{
            'image_id': img_id,
            'bbox': box,
            'score': score,
            'category_id': label
        } for box, score, label in zip(boxes, scores, labels)]
        evaluated.append(img_id)

    evl = cocoeval.COCOeval(data.coco, data.coco.loadRes(results), 'bbox')
    evl.params.imgIds = evaluated
    evl.evaluate()
    evl.accumulate()
    evl.summarize()

    ann_boxes = [ann['bbox'] for ann in anns]
    ann_labels = [ann['category_id'] for ann in anns]
    utils.show(img, boxes, ann_boxes, labels, ann_labels)

@cli.command()
@click.option('--gln', type=click.Choice(['logging', 'plain']), default='logging')
@click.option(
    '--input-sizes', multiple=True, type = (int, int),
    default = [(300, 200), (200, 300), (600, 400)]
)
def gln_build_assistant(gln, input_sizes):
    models = {
        'logging': proposals.state_logging_gln,
        'plain': proposals.gln,
    }
    model = models[gln]()
    loss = model(
        [torch.randn(3, h, w) for w, h in input_sizes],
        [
            {'boxes': torch.Tensor([[0, 0, 1, 1]]), 'labels': torch.tensor([0], dtype=torch.long), 'gaussians': torch.randn(h, w)}
            for w, h in input_sizes
        ]
    )
    print(loss)
    total_loss = loss['classification'] + loss['bbox_regression'] + loss['gaussian']
    total_loss.backward()

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
@click.option('--method', type=click.Choice(['normal', 'kant', 'simple']), default='normal')
@click.option('--tanh/--no-tanh', default=False,)
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=11)
@click.option('--gpus', type=int, default=1)
@click.option('--load', default=None)
@click.option('--trim-module-prefix/--no-trim-module-prefix', default=False)
@click.option('--hyperopt-params/--no-hyperopt-params', default=True)
def train_gln(imgs, annotations, eval_annotations, out_dir, method, tanh, batch_size, dataloader_workers, epochs, gpus, load, trim_module_prefix, hyperopt_params):
    gauss_methods = {
        'normal': {'gauss_generate_method': datautils.generate_via_multivariate_normal, 'gauss_join_method': datautils.join_via_max},
        'kant': {'gauss_generate_method': datautils.generate_via_kant_method, 'gauss_join_method': datautils.join_via_replacement},
        'simple': {'gauss_generate_method': datautils.generate_via_simple_and_scaled, 'gauss_join_method': datautils.join_via_max},
    }
    dataset = datautils.SKU110KDataset(imgs, annotations, skip=SKU110K_SKIP, tanh=tanh, **gauss_methods[method])
    evalset = datautils.SKU110KDataset(imgs, eval_annotations, skip=SKU110K_SKIP, include_gaussians=False)

    options = proposals_training.ProposalTrainingOptions()
    options.dataset = dataset
    options.evalset = evalset
    options.output_path = out_dir
    options.tanh = tanh
    options.gaussian_loss_params = {'tanh': tanh, 'negative_threshold': -1, 'positive_threshold': -0.8} if tanh else {}
    options.load = load
    options.trim_module_prefix = trim_module_prefix
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.gpus = gpus

    if hyperopt_params:
        options.lr_multiplier = 0.995
        options.gaussian_loss_params = {'tanh': tanh, 'negative_threshold': -1, 'positive_threshold': 0.3} if tanh else {'positive_threshold': 0.65}

    args = (options,)
    if gpus > 1:
        utils.ensure_dist_file_clean()
        mp.spawn(proposals_training.train_proposal_generator, args=args, nprocs=gpus)
    else:
        proposals_training.train_proposal_generator(0, options)

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option('--name', type=str, default='gln')
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option('--samples', type=int, default=100)
@click.option('--load/--no-load', default=False)
@click.option('--load-algo', type=click.Path())
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
def hyperopt_gln(imgs, annotations, eval_annotations, name, batch_size, dataloader_workers, epochs, samples, load, load_algo, out_dir):
    config = {
        'tanh': tune.choice([True, False]),

        'multiplier': tune.uniform(0.8, 0.99999),

        'scale_class': tune.uniform(0.1, 10),
        'scale_gaussian': tune.uniform(0.1, 100),

        'gauss_loss_neg_thresh': 0,
        'gauss_loss_pos_thresh': tune.uniform(0, 1),
    }

    initial_configs = [
        {
            'tanh': True,
            'multiplier': 0.99,
            'scale_class': 1,
            'scale_gaussian': 1,
            'gauss_loss_neg_thresh': 0,
            'gauss_loss_pos_thresh': 0.1,
        },
        {
            'tanh': False,
            'multiplier': 0.99,
            'scale_class': 1,
            'scale_gaussian': 1,
            'gauss_loss_neg_thresh': 0,
            'gauss_loss_pos_thresh': 0.1,
        },
    ]

    algo = HyperOptSearch(points_to_evaluate=initial_configs if not load and load_algo is None else None)
    if load_algo is not None:
        algo.restore(load_algo)

    scheduler = ASHAScheduler(max_t = epochs, grace_period = 2)
    result = tune.run(
        partial(hyperopt.gln,
            imgs=imgs, annotations=annotations, eval_annotations=eval_annotations, skip=SKU110K_SKIP,
            batch_size=batch_size, dataloader_workers=dataloader_workers, epochs=epochs),
        name=name,
        metric='average_precision',
        mode='max',
        resources_per_trial={'gpu': 1, 'cpu': dataloader_workers + 1},
        max_failures=2, # Single-GPU training of GLN is prone to exploding gradients
        raise_on_failed_trial=False,
        config=config,
        num_samples=samples,
        scheduler=scheduler,
        search_alg=algo,
        resume=load,
    )
    algo.save(os.path.join(out_dir, f'{name}_search.pkl'))
    df = result.results_df
    for tanh in (True, False):
        matching = df[df['config.tanh'] == tanh]
        print(f'Best with tanh={tanh}: {matching.loc[matching["average_precision"].idxmax()]}')
        print()

@cli.command()
@click.option('--dataset', type=click.Choice(('sku110k', 'gp180', 'gpbaseline')), default='sku110k')
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--metric-workers', type=int, default=8)
@click.option('--iou-threshold', '-t', type=float, multiple=True, default=(0.5,))
@click.option('--coco/--no-coco', default=False)
@click.option('--trim-module-prefix/--no-trim-module-prefix', default=False)
@click.option('--plots/--no-plots', default=True)
@click.option('--plot-res-reduction', type=int, default=200)
@click.argument('state-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def eval_gln(dataset, imgs, annotations, batch_size, dataloader_workers, metric_workers, iou_threshold, coco, trim_module_prefix, plots, plot_res_reduction, state_file):
    if dataset == 'sku110k':
        dataset = datautils.SKU110KDataset(imgs, annotations, skip=SKU110K_SKIP, include_gaussians=False, flip_chance=0)
    elif dataset == 'gp180':
        dataset = datautils.GroceryProductsTestSet(imgs, annotations, retinanet_annotations=True)
    else:
        dataset = datautils.GPBaselineDataset(imgs, annotations)

    thresholds = [f.item() for f in torch.linspace(.5, .95, 10)] if coco else iou_threshold
    evaluation = proposals_eval.evaluate_gln(state_file, dataset, thresholds=thresholds,
        batch_size=batch_size, num_workers=dataloader_workers, num_metric_processes=metric_workers, trim_module_prefix=trim_module_prefix,
        plots=plots, resolution_reduction=plot_res_reduction)
    ap = 0
    ar = 0
    for t in thresholds:
        print(f'{t}:\t{evaluation[t]}')
        ap += evaluation[t]['ap']
        ar += evaluation[t]['ar_300']
    print(f'--> AP {ap / len(thresholds)}')
    print(f'--> AR300 {ar / len(thresholds)}')

@cli.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option('--outlier-threshold', type=int, default=3)
@click.option('--trim-module-prefix/--no-trim-module-prefix', default=False)
@click.argument('state-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def seek_sku110k_outliers(imgs, annotations, outlier_threshold, trim_module_prefix, state_file):
    def outliers(loss):
        is_outlier = loss > loss.mean() + outlier_threshold * loss.std()
        indices = is_outlier.nonzero(as_tuple=True)[0]
        return set(indices)

    dataset = datautils.SKU110KDataset(imgs, annotations, skip=SKU110K_SKIP)

    state_dict = torch.load(state_file)[proposals_training.MODEL_STATE_DICT_KEY]
    if trim_module_prefix:
        state_dict = utils.trim_module_prefix(state_dict)

    model = proposals.gln().cuda()
    model.load_state_dict(state_dict)

    class_loss = torch.zeros(len(dataset), dtype=torch.float)
    reg_loss = torch.zeros(len(dataset), dtype=torch.float)
    gauss_loss = torch.zeros(len(dataset), dtype=torch.float)
    with torch.no_grad():
        for i, (img, target) in enumerate(dataset):
            if i % 200 == 0:
                print(f'{i}...')
            img = img.cuda()[None]
            target = [{k: v.cuda() if k in ['boxes', 'labels', 'gaussians'] else v for k, v in target.items()}]
            loss = model(img, target)
            class_loss[i] = loss['classification']
            reg_loss[i] = loss['bbox_regression']
            gauss_loss[i] = loss['gaussian']

    outlier_indices = outliers(class_loss) | outliers(reg_loss) | outliers(gauss_loss)

    print('-'*10)
    print(f'Mean loss:\t{class_loss.mean()}\t{reg_loss.mean()}\t{gauss_loss.mean()}')
    print(f'Std loss:\t{class_loss.std()}\t{reg_loss.std()}\t{gauss_loss.std()}')
    print('-'*10)

    if len(outlier_indices) == 0:
        print('No outliers! Have a nice day!')
    for i in outlier_indices:
        _, entry = dataset[i]
        print(f'Outlier at index {i}: {entry["image_name"]}')
        print(f'\t{class_loss[i]}\t{reg_loss[i]}\t{gauss_loss[i]}')

@cli.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
@click.option('--batch-size', type=int, default=16)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=1)
@click.option('--masks/--no-masks', default=False)
def pretrain_cls_gan(source_dir, target_imgs, target_annotations, out_dir, batch_size, dataloader_workers, epochs, masks):
    options = classification_training.ClassificationTrainingOptions()

    options.dataset = datautils.GroceryProductsDataset(source_dir, include_masks=masks)
    options.discriminatorset = datautils.TargetDomainDataset(target_imgs, target_annotations, skip=SKU110K_SKIP)
    options.output_path = out_dir
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.masks = masks

    classification_training.pretrain_gan(options)

@cli.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--source-type', type=click.Choice(('gp', 'internal')), default='gp')
@click.option('--only', type=str, multiple=True)
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--eval-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--eval-data',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
@click.option('--batch-norm/--no-batch-norm', default=False)
@click.option('--masks/--no-masks', default=False)
@click.option('--batch-size', type=int, default=4)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option('--load-gan', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=PRETRAINED_GAN_FILE)
@click.option('--load-enc', default=None)
@click.option('--gpus', type=int, default=1)
@click.option('--hyperopt-params/--no-hyperopt-params', default=True)
def train_dihe(source_dir, source_type, only, target_imgs, target_annotations, eval_imgs, eval_annotations, eval_data, out_dir, batch_norm, masks, batch_size, dataloader_workers, epochs, load_gan, load_enc, gpus, hyperopt_params):
    options = classification_training.ClassificationTrainingOptions()

    if source_type == 'gp':
        options.dataset = datautils.GroceryProductsDataset(source_dir, include_annotations=True, include_masks=masks, only=only if len(only) else None)
    else:
        options.dataset = datautils.InternalTrainSet(source_dir[0], include_annotations=True, include_masks=masks)
        options.evaldata = datautils.GroceryProductsDataset(eval_data, include_annotations=True, include_masks=masks, only=only if len(only) else None)
    options.discriminatorset = datautils.TargetDomainDataset(target_imgs, target_annotations, skip=SKU110K_SKIP)
    options.evalset = datautils.GroceryProductsTestSet(eval_imgs, eval_annotations, only=GP_TEST_VALIDATION_SET_SIZE)

    options.load_gan = load_gan
    options.load_encoder = load_enc
    options.output_path = out_dir
    options.batchnorm = batch_norm
    options.masks = masks
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.gpus = gpus

    if hyperopt_params:
        options.enc_lr = 8e-7
        options.enc_multiplier = 0.9

    if gpus > 1:
        utils.ensure_dist_file_clean()
        mp.spawn(classification_training.train_dihe, args=(options,), nprocs=gpus)
    else:
        classification_training.train_dihe(0, options)

@cli.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--only', type=str, multiple=True)
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE
)
@click.option(
    '--eval-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--masks/--no-masks', default=False)
@click.option('--batch-size', type=int, default=4)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option('--samples', type=int, default=100)
@click.option('--name', type=str, default='dihe')
@click.option('--load-gan', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=PRETRAINED_GAN_FILE)
@click.option('--load/--no-load', default=False)
@click.option('--load-algo', type=click.Path())
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
def hyperopt_dihe(source_dir, only, target_imgs, target_annotations, eval_imgs, eval_annotations, masks, batch_size, dataloader_workers, epochs, samples, name, load_gan, load, load_algo, out_dir):
    config = {
        'batchnorm': tune.choice([True, False]),
        'multiplier': tune.uniform(0.5, 0.99999),
        'enc_lr': tune.uniform(1e-9, 1e-3),
    }

    algo = HyperOptSearch()
    if load_algo is not None:
        algo.restore(load_algo)

    scheduler = ASHAScheduler(max_t = epochs)
    result = tune.run(
        partial(hyperopt.dihe,
            source_dir=source_dir, target_imgs=target_imgs, target_annotations=target_annotations, eval_imgs=eval_imgs, eval_annotations=eval_annotations,
            load_gan=load_gan, masks=masks, source_only=only, target_skip=SKU110K_SKIP, eval_only=GP_TEST_VALIDATION_SET_SIZE,
            batch_size=batch_size, dataloader_workers=dataloader_workers, epochs=epochs),
        name=name,
        metric='accuracy',
        mode='max',
        resources_per_trial={'gpu': 1, 'cpu': dataloader_workers + 1},
        config=config,
        num_samples=samples,
        scheduler=scheduler,
        search_alg=algo,
        resume=load,
    )
    algo.save(os.path.join(out_dir, f'{name}_search.pkl'))
    df = result.results_df
    for batchnorm in (True, False):
        matching = df[df['config.batchnorm'] == batchnorm]
        print(f'Best with batchnorm={batchnorm}: {matching.loc[matching["accuracy"].idxmax()]}')
        print()

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--model', type=click.Choice(('vgg16', 'resnet50')), default='vgg16')
@click.option('--resnet-layers', type=int, multiple=True, default=[2, 3])
@click.option('--batch-norm/--no-batch-norm', default=True)
@click.option('--batch-size', type=int, default=8)
@click.option('--dataloader-workers', type=int, default=8)
@click.option('--enc-weights')
@click.option('--only', type=click.Choice(('none', 'test', 'val')), default='none')
@click.option('--knn', type=int, multiple=True, default=(1,))
def eval_dihe(img_dir, test_imgs, annotations, model, resnet_layers, batch_norm, batch_size, dataloader_workers, enc_weights, only, knn):
    sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)

    only_list = None
    skip_list = None
    if only == 'test':
        skip_list = GP_TEST_VALIDATION_SET_SIZE
    elif only == 'val':
        only_list = GP_TEST_VALIDATION_SET_SIZE
    testset = datautils.GroceryProductsTestSet(test_imgs, annotations, only=only_list, skip=skip_list)

    if model == 'vgg16':
        encoder = classification.macvgg_embedder(model='vgg16_bn' if batch_norm else 'vgg16', pretrained=enc_weights is None).cuda()
    elif model == 'resnet50':
        encoder = classification.macresnet_encoder(pretrained=enc_weights is None, desc_layers=resnet_layers).cuda()
    if enc_weights is not None:
        state = torch.load(enc_weights)
        encoder.load_state_dict(state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()

    classification_eval.eval_dihe(encoder, sampleset, testset, batch_size, dataloader_workers, k=knn)

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--resnet-layers', type=int, multiple=True, default=[2, 3])
@click.option('--batch-norm/--no-batch-norm', default=False)
@click.option('--batch-size', type=int, default=8)
@click.option('--dataloader-workers', type=int, default=8)
@click.option('--enc-weights')
@click.option('--only', type=click.Choice(('none', 'test', 'val')), default='none')
@click.option('--knn', type=int, default=4)
@click.option('--load-classifier-index', type=click.Path())
def visualize_classification_performance(img_dir, test_imgs, annotations, resnet_layers, batch_norm, batch_size, dataloader_workers, enc_weights, only, knn, load_classifier_index):
    sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
    rebuildset = datautils.GroceryProductsDataset(img_dir, include_annotations=True, resize=False)

    only_list = None
    skip_list = None
    if only == 'test':
        skip_list = GP_TEST_VALIDATION_SET_SIZE
    elif only == 'val':
        only_list = GP_TEST_VALIDATION_SET_SIZE
    testset = datautils.GroceryProductsTestSet(test_imgs, annotations, only=only_list, skip=skip_list)

    encoder = classification.macvgg_embedder(model='vgg16_bn' if batch_norm else 'vgg16', pretrained=enc_weights is None).cuda()
    if enc_weights is not None:
        state = torch.load(enc_weights)
        encoder.load_state_dict(state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()

    classifier = production.Classifier(encoder, sampleset, k=knn, batch_size=8, load=load_classifier_index)

    fig = plt.figure(figsize=(5.5, 5.5))
    fig.set_dpi(200)
    gs = fig.add_gridspec(knn+1, knn+1)
    source_ax = [fig.add_subplot(gs[y, 0]) for y in range(knn+1)]
    target_ax = [[fig.add_subplot(gs[y, x]) for x in range(1, knn+1)] for y in range(knn+1)]
    for s_ax, t_ax in zip(source_ax, target_ax):
        img, anns, boxes = random.choice(testset)
        box = random.choice(boxes).to(dtype=torch.long)
        img = img[:, max(0, box[1]):box[3], max(0, box[0]):box[2]]
        utils.build_fig(img, ax=s_ax)

        anns = classifier.classify(datautils.resize_for_classification(img)[None])[0]
        for ann, ax in zip(anns, t_ax):
            idx = rebuildset.index_for_ann(ann)
            im, _, _, _ = rebuildset[idx]
            utils.build_fig(im, ax=ax)

    plt.show()

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option('--datatype', type=click.Choice(('gp', 'internal')), default='gp')
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def prebuild_classifier_index(img_dir, datatype, out_dir, dihe_state):
    if datatype == 'gp':
        sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
    else:
        sampleset = datautils.InternalTrainSet(img_dir[0], include_annotations=True)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    del enc_state

    classifier = production.Classifier(encoder, sampleset, verbose=True)
    classifier.save_index(os.path.join(out_dir, 'classifier_index.pkl'))

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option('--iou-threshold', '-t', type=float, multiple=True, default=(0.5,))
@click.option('--coco/--no-coco', default=False)
@click.option('--load-classifier-index', type=click.Path())
@click.argument('gln-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def eval_product_detection(img_dir, test_imgs, annotations, iou_threshold, coco, load_classifier_index, gln_state, dihe_state):
    sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
    testset = datautils.GroceryProductsTestSet(test_imgs, annotations, retinanet_annotations=True)

    thresholds = [f.item() for f in torch.linspace(.5, .95, 10)] if coco else iou_threshold

    proposal_generator = proposals_eval.load_gln(gln_state, False, detections_per_img=200)
    proposal_generator.requires_grad_(False)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    del enc_state

    res, all_res = detection_eval.evaluate_detections(proposal_generator, encoder, testset, sampleset, thresholds=thresholds, load_classifier_index=load_classifier_index)

    mam = detection_eval.mean_average_metrics(res, thresholds)
    m_ap = 0
    m_ar = 0
    for t in thresholds:
        print(t, all_res[t])
        print(t, mam[t])
        m_ap += mam[t]['map']
        m_ar += mam[t]['mar300']
    print(f'--> mAP {m_ap / len(thresholds)}')
    print(f'--> mAR300 {m_ar / len(thresholds)}')

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--test-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR
)
@click.option('--datatype', type=click.Choice(('gp', 'internal')), default='gp')
@click.option('--load-classifier-index', type=click.Path())
@click.option('--plano-idx', type=int)
@click.argument('gln-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def rebuild_scene(img_dir, test_imgs, test_annotations, planograms, datatype, load_classifier_index, plano_idx, gln_state, dihe_state):
    if datatype == 'gp':
        planoset = datautils.PlanogramTestSet(test_imgs, test_annotations, planograms)
        sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
        rebuildset = datautils.GroceryProductsDataset(img_dir, include_annotations=True, resize=False)
    else:
        planoset = datautils.InternalPlanoSet(planograms)
        sampleset = datautils.InternalTrainSet(img_dir[0], include_annotations=True)
        rebuildset = datautils.InternalTrainSet(img_dir[0], include_annotations=True, resize=False)

    proposal_generator = proposals_eval.load_gln(gln_state, False, detections_per_img=200)
    proposal_generator.requires_grad_(False)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    del enc_state

    generator = production.ProposalGenerator(proposal_generator, confidence_threshold=0.5)
    classifier = production.Classifier(encoder, sampleset, batch_size=8, load=load_classifier_index)

    datum = planoset[plano_idx] if plano_idx is not None else random.choice(planoset)
    if datatype == 'gp':
        image, _, _, plano = datum
    else:
        image, plano = datum
    boxes, images = generator.generate_proposals_and_images(image)
    classes = [ann[0] for ann in classifier.classify(images)]

    maxy = boxes[:, 3].max().item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    utils.build_fig(image, detections=tvops.box_convert(boxes, 'xyxy', 'xywh'), ax=ax1)
    utils.build_rebuild(boxes, classes, rebuildset, maxy, ax=ax2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    boxes = plano['boxes']
    labels = [l.split('.')[0] for l in plano['labels']] if datatype == 'gp' else plano['labels']
    maxy = boxes[:, 3].max().item()
    utils.build_fig(image, ax=ax1)
    utils.build_rebuild(boxes, labels, rebuildset, maxy, ax=ax2)
    plt.show()

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--test-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--planograms',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR
)
@click.option('--datatype', type=click.Choice(('gp', 'internal')), default='gp')
@click.option('--load-classifier-index', type=click.Path())
@click.option('--verbose/--no-verbose', default=False)
@click.argument('gln-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def eval_planograms(img_dir, test_imgs, test_annotations, planograms, datatype, load_classifier_index, verbose, gln_state, dihe_state):
    if datatype == 'gp':
        planoset = datautils.PlanogramTestSet(test_imgs, test_annotations, planograms)
        sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
    else:
        planoset = datautils.InternalPlanoSet(planograms)
        sampleset = datautils.InternalTrainSet(img_dir[0], include_annotations=True)

    proposal_generator = proposals_eval.load_gln(gln_state, False)
    proposal_generator.requires_grad_(False)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    del enc_state

    generator = production.ProposalGenerator(proposal_generator)
    classifier = production.Classifier(encoder, sampleset, batch_size=8, load=load_classifier_index)
    comparator = production.PlanogramComparator()

    evaluator = production.PlanogramEvaluator(generator, classifier, comparator)
    total_a = 0
    total_e = 0
    for i, (datum) in enumerate(planoset):
        if datatype == 'gp':
            img, _, _, plano = datum
        else:
            img, plano = datum

        acc = evaluator.evaluate(img, plano)
        err = acc - plano['actual_accuracy']
        sqerr = err ** 2
        if verbose:
            print(f'Detected accuracy: {acc:.3f}, Actual accuracy: {plano["actual_accuracy"]:.3f}, Error: {err:.3f}, SE: {sqerr:.3f}')
        elif i % 10 == 0:
            print(i)
        total_e += sqerr
        total_a += acc
    print(f'--> Mean accuracy {(total_a / len(planoset)).item()}')
    print(f'--> MSE: {(total_e / len(planoset)).item()}')

@cli.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR
)
@click.option(
    '--test-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR
)
@click.option(
    '--planos',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_PLANO_DIR
)
@click.option('--datatype', type=click.Choice(('gp', 'internal')), default='gp')
@click.option('--load-classifier-index', type=click.Path())
@click.option('--plano-idx', type=int)
@click.argument('gln-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def plot_planogram_eval(img_dir, test_imgs, test_annotations, planos, datatype, load_classifier_index, plano_idx, gln_state, dihe_state):
    if datatype == 'gp':
        planoset = datautils.PlanogramTestSet(test_imgs, test_annotations, planos)
        sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)
        rebuildset = datautils.GroceryProductsDataset(img_dir, include_annotations=True, resize=False)
    else:
        planoset = datautils.InternalPlanoSet(planos)
        sampleset = datautils.InternalTrainSet(img_dir[0], include_annotations=True)
        rebuildset = datautils.InternalTrainSet(img_dir[0], include_annotations=True, resize=False)

    proposal_generator = proposals_eval.load_gln(gln_state, False)
    proposal_generator.requires_grad_(False)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[classification_training.EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    del enc_state

    datum = planoset[plano_idx] if plano_idx is not None else random.choice(planoset)
    if datatype == 'gp':
        image, _, _, expected = datum
    else:
        image, expected = datum
    generator = production.ProposalGenerator(proposal_generator)
    classifier = production.Classifier(encoder, sampleset, batch_size=8, load=load_classifier_index)

    boxes, images = generator.generate_proposals_and_images(image)
    classes = [ann[0] for ann in classifier.classify(images)]
    actual = {'boxes': boxes.detach().cpu(), 'labels': classes}

    h, w = image.shape[1:]
    reproj_threshold = min(h, w) * 0.01

    maxy = boxes[:, 3].max().item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    utils.build_fig(image, detections=tvops.box_convert(boxes, 'xyxy', 'xywh'), ax=ax1)
    utils.build_rebuild(boxes, classes, rebuildset, maxy, ax=ax2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    ge = expected['graph'] if 'graph' in expected else planograms.build_graph(expected['boxes'], expected['labels'], 0.5)
    ga = planograms.build_graph(actual['boxes'], actual['labels'], 0.5)
    utils.build_rebuild(expected['boxes'], expected['labels'], rebuildset, ax=ax1)
    utils.draw_planograph(ge, expected['boxes'], ax=ax1, flip_y=True)
    utils.build_rebuild(boxes, classes, rebuildset, maxy, ax=ax2)
    utils.draw_planograph(ga, actual['boxes'], ax=ax2, flip_y=True)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    matching = planograms.large_common_subgraph(ge, ga)
    nodes_e, nodes_a = (list(l) for l in zip(*matching)) if len(matching) else ([],[])
    sge = ge.subgraph(nodes_e)
    sga = ga.subgraph(nodes_a)
    utils.build_rebuild(expected['boxes'], expected['labels'], rebuildset, ax=ax1)
    utils.draw_planograph(sge, expected['boxes'], ax=ax1, flip_y=True)
    utils.build_rebuild(boxes, classes, rebuildset, maxy, ax=ax2)
    utils.draw_planograph(sga, actual['boxes'], ax=ax2, flip_y=True)
    if not len(matching):
        plt.show()
        return

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12)) if image.shape[2] < image.shape[1] else plt.subplots(2, 1, figsize=(12, 12))
    found, found_actual, missing_indices, missing_positions, missing_labels = planograms.finalize_via_ransac(
        matching, expected['boxes'], actual['boxes'], expected['labels'], actual['labels'],
        reproj_threshold=reproj_threshold, return_matched_actual=True
    )
    missing_positions = tvops.clip_boxes_to_image(missing_positions, image.shape[1:])
    valid_positions = (missing_positions[:,2] - missing_positions[:,0] > 1) & (missing_positions[:,3] - missing_positions[:,1] > 1)
    missing_indices = missing_indices[valid_positions]
    missing_positions = missing_positions[valid_positions]
    missing_labels = [l for l, v in zip(missing_labels, valid_positions) if v]

    if len(missing_positions) > 0:
        found_round2 = torch.full((len(missing_indices),), False)
        missing_imgs = torch.stack([datautils.resize_for_classification(image[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in missing_positions.to(dtype=torch.long)])
        reclass_labels = classifier.classify(missing_imgs)
        for idx, (expected_label, actual_label) in enumerate(zip(missing_labels, reclass_labels)):
            if expected_label == actual_label[0]:
                found_round2[idx] = True
    utils.build_fig(image,
        groundtruth=tvops.box_convert(actual['boxes'][found_actual], 'xyxy', 'xywh'),
        detections=tvops.box_convert(missing_positions, 'xyxy', 'xywh'),
    )
    if len(missing_positions) > 0: utils.plot_boxes(tvops.box_convert(missing_positions[found_round2], 'xyxy', 'xywh'), color='yellow', hl_color='orange')

    plt.show()

if __name__ == '__main__':
    cli()
