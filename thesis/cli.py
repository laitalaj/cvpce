import os
import random
import re
import shutil
from functools import partial

import click
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
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from . import datautils
from . import utils
from . import proposals_training, proposals_eval
from . import classification_training, classification_eval
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
    utils.rel_path(*DATA_DIR, 'Planogram Dataset', 'extra_products'),
)
'''
GP_TEST_DIR = utils.rel_path(*GP_ROOT, 'Testing')
GP_ANN_DIR = utils.rel_path(*DATA_DIR, 'Planogram Dataset', 'annotations')
GP_PLANO_DIR = utils.rel_path(*DATA_DIR, 'Planogram Dataset', 'planograms')
GP_TEST_VALIDATION_SET = ['s1_15.csv', 's2_3.csv', 's2_30.csv', 's2_143.csv', 's2_157.csv', 's3_111.csv', 's3_260.csv', 's5_55.csv']
GP_PLANO_VALIDATION_SET = [f'{s.split(".")[0]}.json' for s in GP_TEST_VALIDATION_SET]

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
    type=click.Choice(['normal', 'kant']),
    default='normal'
)
@click.option('--flip/--no-flip', default=False)
@click.option('--gaussians/--no-gaussians', default=True)
@click.option('--model', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.option('--conf-thresh', type=float, default=0.45)
@click.option('--save', type=click.Path(writable=True))
def visualize_sku110k(imgs, annotations, index, method, flip, gaussians, model, conf_thresh, save):
    gauss_methods = {
        'normal': {'gauss_generate_method': datautils.generate_via_multivariate_normal, 'gauss_join_method': datautils.join_via_max},
        'kant': {'gauss_generate_method': datautils.generate_via_kant_method, 'gauss_join_method': datautils.join_via_replacement},
    }
    data = datautils.SKU110KDataset(imgs, annotations, flip_chance=0, **gauss_methods[method])
    if index is None:
        img, anns = random.choice(data)
    else:
        if isinstance(index, str):
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
        groundtruth=[[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in anns['boxes']]
    ) # TODO: Torch has a built in function for converting between coordinate systems
    if save is not None:
        utils.save(img, save, groundtruth=[[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in anns['boxes']])
    if gaussians: utils.show(anns['gaussians'])

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
@click.option('--only', type=click.Choice(('none', 'test', 'val')), default='none')
def gp_test_distribution(imgs, annotations, only):
    only_list = None
    skip_list = None
    if only == 'test':
        skip_list = GP_TEST_VALIDATION_SET
    elif only == 'val':
        only_list = GP_TEST_VALIDATION_SET

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
@click.option('--method', type=click.Choice(['normal', 'kant']), default='normal')
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=11)
@click.option('--gpus', type=int, default=1)
@click.option('--load', default=None)
@click.option('--trim-module-prefix/--no-trim-module-prefix', default=False)
def train_gln(imgs, annotations, eval_annotations, out_dir, method, batch_size, dataloader_workers, epochs, gpus, load, trim_module_prefix):
    gauss_methods = {
        'normal': {'gauss_generate_method': datautils.generate_via_multivariate_normal, 'gauss_join_method': datautils.join_via_max},
        'kant': {'gauss_generate_method': datautils.generate_via_kant_method, 'gauss_join_method': datautils.join_via_replacement},
    }
    dataset = datautils.SKU110KDataset(imgs, annotations, skip=SKU110K_SKIP, **gauss_methods[method])
    evalset = datautils.SKU110KDataset(imgs, eval_annotations, skip=SKU110K_SKIP, include_gaussians=False)

    options = proposals_training.ProposalTrainingOptions()
    options.dataset = dataset
    options.evalset = evalset
    options.output_path = out_dir
    options.load = load
    options.trim_module_prefix = trim_module_prefix
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.gpus = gpus

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
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option('--samples', type=int, default=100)
def hyperopt_gln(imgs, annotations, eval_annotations, batch_size, dataloader_workers, epochs, samples):
    config = {
        'lr': tune.uniform(0.001, 0.1),
        'decay': tune.uniform(0.00001, 0.01),
        'momentum': tune.uniform(0.7, 0.999),
        'multiplier': tune.uniform(0.8, 0.99999),
        'scale_class': tune.uniform(0.1, 10),
        'scale_bbox': tune.uniform(0.1, 10),
        'scale_gaussian': tune.uniform(0.1, 100)
    }
    algo = TuneBOHB(max_concurrent=4, metric='ap', mode='max')
    scheduler = HyperBandForBOHB(metric='ap', mode='max', max_t=epochs)
    result = tune.run(
        partial(hyperopt.gln,
            imgs=imgs, annotations=annotations, eval_annotations=eval_annotations, skip=SKU110K_SKIP,
            batch_size=batch_size, dataloader_workers=dataloader_workers, epochs=epochs),
        resources_per_trial={'gpu': 1},
        config=config,
        num_samples=samples,
        scheduler=scheduler,
        search_alg=algo,
    )
    best = result.get_best_trial('ap', 'max')
    print(f'Best: Config: {best.config}, AP: {best.last_result["ap"]}')

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
@click.option('--batch-size', type=int, default=1)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--metric-workers', type=int, default=8)
@click.option('--iou-threshold', '-t', type=float, multiple=True, default=(0.5,))
@click.option('--coco/--no-coco', default=False)
@click.option('--trim-module-prefix/--no-trim-module-prefix', default=False)
@click.option('--plots/--no-plots', default=True)
@click.argument('state-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def eval_gln(imgs, annotations, batch_size, dataloader_workers, metric_workers, iou_threshold, coco, trim_module_prefix, plots, state_file):
    dataset = datautils.SKU110KDataset(imgs, annotations, skip=SKU110K_SKIP, include_gaussians=False)
    thresholds = torch.linspace(.5, .95, 10) if coco else iou_threshold
    evaluation = proposals_eval.evaluate_gln(state_file, dataset, thresholds=thresholds,
        batch_size=batch_size, num_workers=dataloader_workers, num_metric_processes=metric_workers, trim_module_prefix=trim_module_prefix, plots=plots)
    for t in iou_threshold:
        print(f'{t}:\t{evaluation[t]}')

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
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR
)
@click.option('--batch-norm/--no-batch-norm', default=True)
@click.option('--masks/--no-masks', default=False)
@click.option('--batch-size', type=int, default=4)
@click.option('--dataloader-workers', type=int, default=4)
@click.option('--epochs', type=int, default=10)
@click.option('--load-gan', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=PRETRAINED_GAN_FILE)
@click.option('--load-enc', default=None)
@click.option('--gpus', type=int, default=1)
def train_dihe(source_dir, only, target_imgs, target_annotations, eval_imgs, eval_annotations, out_dir, batch_norm, masks, batch_size, dataloader_workers, epochs, load_gan, load_enc, gpus):
    options = classification_training.ClassificationTrainingOptions()

    options.dataset = datautils.GroceryProductsDataset(source_dir, include_annotations=True, include_masks=masks, only=only if len(only) else None)
    options.discriminatorset = datautils.TargetDomainDataset(target_imgs, target_annotations, skip=SKU110K_SKIP)
    options.evalset = datautils.GroceryProductsTestSet(eval_imgs, eval_annotations, only=GP_TEST_VALIDATION_SET)

    options.load_gan = load_gan
    options.load_encoder = load_enc
    options.output_path = out_dir
    options.batchnorm = batch_norm
    options.masks = masks
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.gpus = gpus

    if gpus > 1:
        utils.ensure_dist_file_clean()
        mp.spawn(classification_training.train_dihe, args=(options,), nprocs=gpus)
    else:
        classification_training.train_dihe(0, options)

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
@click.option('--knn', default=1)
def eval_dihe(img_dir, test_imgs, annotations, model, resnet_layers, batch_norm, batch_size, dataloader_workers, enc_weights, only, knn):
    sampleset = datautils.GroceryProductsDataset(img_dir, include_annotations=True)

    only_list = None
    skip_list = None
    if only == 'test':
        skip_list = GP_TEST_VALIDATION_SET
    elif only == 'val':
        only_list = GP_TEST_VALIDATION_SET
    testset = datautils.GroceryProductsTestSet(test_imgs, annotations, only=only_list, skip=skip_list)

    if model == 'vgg16':
        encoder = classification.macvgg_embedder(model='vgg16_bn' if batch_norm else 'vgg16', pretrained=enc_weights is None).cuda()
    elif model == 'resnet50':
        encoder = classification.macresnet_encoder(pretrained=enc_weights is None, desc_layers=resnet_layers).cuda()
    encoder.eval()
    if enc_weights is not None:
        state = torch.load(enc_weights)
        encoder.load_state_dict(state[classification_training.EMBEDDER_STATE_DICT_KEY])

    classification_eval.eval_dihe(encoder, sampleset, testset, batch_size, dataloader_workers, k=knn)


if __name__ == '__main__':
    cli()
