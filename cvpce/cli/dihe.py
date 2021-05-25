import os
import random
from functools import partial

import click
import torch
from matplotlib import pyplot as plt
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from torch import multiprocessing as mp

from .. import classification_training, classification_eval, datautils, production, utils
from ..defaults import (GP_TRAIN_FOLDERS, GP_TEST_DIR, GP_ANN_DIR, GP_TEST_VALIDATION_SET_SIZE,
    SKU110K_IMG_DIR, SKU110K_ANNOTATION_FILE, SKU110K_SKIP,
    OUT_DIR, PRETRAINED_GAN_FILE)
from ..models import classification

@click.group()
def dihe():
    '''
    DIHE training and evaluation.

    This command group contains functionality related to the Domain Invariant Hierarchial Embedding (DIHE)
    of Tonioni et al. (2019).
    It especially includes functions for training a DIHE, evaluating retail product classification performance
    and for visualizing said performance.
    '''
    pass

@dihe.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to GP training image root'
)
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR, show_default=True,
    help='Path to SKU-110K image root for discriminator training'
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE, show_default=True,
    help='Path to SKU-110K annotation file for discriminator training'
)
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR, show_default=True,
    help='Output directory for models and images'
)
@click.option('--batch-size', type=int, default=16, show_default=True, help='Batch size')
@click.option('--dataloader-workers', type=int, default=4, show_default=True, help='Number of data loading processes')
@click.option('--epochs', type=int, default=1, show_default=True, help='Number of epochs to train')
@click.option('--masks/--no-masks', default=False, show_default=True, help='Use/don\'t use product masks when training generator')
def pretrain_gan(source_dir, target_imgs, target_annotations, out_dir, batch_size, dataloader_workers, epochs, masks):
    '''
    Pretrain the GAN for DIHE.
    '''
    options = classification_training.ClassificationTrainingOptions()

    options.dataset = datautils.GroceryProductsDataset(source_dir, include_masks=masks)
    options.discriminatorset = datautils.TargetDomainDataset(target_imgs, target_annotations, skip=SKU110K_SKIP)
    options.output_path = out_dir
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs
    options.masks = masks

    classification_training.pretrain_gan(options)

@dihe.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to training image root'
)
@click.option('--source-type', type=click.Choice(('gp', 'internal')), default='gp', show_default=True,
    help='Training dataset type; gp for Grocery Products, internal for our internal dataset')
@click.option('--only', type=str, multiple=True, help='Only train with given training data groups')
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR, show_default=True,
    help='Path to SKU-110K image root for discriminator training'
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE, show_default=True,
    help='Path to SKU-110K annotation file for discriminator training'
)
@click.option(
    '--eval-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True,
    help='Path to GP test images root for evaluation'
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True,
    help='Path to GP-180 annotations root for evaluation'
)
@click.option(
    '--eval-data',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to Grocery Products training images for evaluation'
)
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR, show_default=True,
    help='Output directory for models and images'
)
@click.option('--batch-norm/--no-batch-norm', default=False, show_default=True, help='Use/don\'t use batch normalization in encoder')
@click.option('--masks/--no-masks', default=False, show_default=True, help='Use/don\'t use product masks in generator')
@click.option('--batch-size', type=int, default=4, show_default=True, help='Batch size per GPU')
@click.option('--dataloader-workers', type=int, default=4, show_default=True, help='Number of data loading processes per GPU')
@click.option('--epochs', type=int, default=10, show_default=True, help='Number of epochs to train')
@click.option('--load-gan', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=PRETRAINED_GAN_FILE,
    show_default=True, help='Path to pretrained GAN state')
@click.option('--load-enc', default=None, help='Path to a saved model to continue training')
@click.option('--gpus', type=int, default=1, show_default=True, help='Number of GPUs to use')
@click.option('--hyperopt-params/--no-hyperopt-params', default=True, show_default=True, help='Use / don\'t use our hyperoptimized parameters')
def train(source_dir, source_type, only, target_imgs, target_annotations, eval_imgs, eval_annotations, eval_data, out_dir, batch_norm, masks, batch_size, dataloader_workers, epochs, load_gan, load_enc, gpus, hyperopt_params):
    '''
    Train DIHE.
    '''
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

@dihe.command()
@click.option(
    '--source-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to training image root'
)
@click.option('--source-type', type=click.Choice(('gp', 'internal')), default='gp', show_default=True,
    help='Training dataset type; gp for Grocery Products, internal for our internal dataset')
@click.option('--only', type=str, multiple=True, help='Only train with given training data groups')
@click.option(
    '--target-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=SKU110K_IMG_DIR, show_default=True,
    help='Path to SKU-110K image root for discriminator training'
)
@click.option(
    '--target-annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=SKU110K_ANNOTATION_FILE, show_default=True,
    help='Path to SKU-110K annotation file for discriminator training'
)
@click.option(
    '--eval-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True,
    help='Path to GP test images root for evaluation'
)
@click.option(
    '--eval-annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True,
    help='Path to GP-180 annotations root for evaluation'
)
@click.option('--masks/--no-masks', default=False, show_default=True, help='Use/don\'t use product masks in generator')
@click.option('--batch-size', type=int, default=4, show_default=True, help='Batch size per model')
@click.option('--dataloader-workers', type=int, default=4, show_default=True, help='Number of data loading processes per model')
@click.option('--epochs', type=int, default=10, show_default=True, help='Number of epochs to train each model')
@click.option('--samples', type=int, default=100, show_default=True, help='Number of models to train')
@click.option('--name', type=str, default='dihe', show_default=True, help='Name for the Ray Tune hyperopt operation')
@click.option('--load-gan', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True), default=PRETRAINED_GAN_FILE,
    show_default=True, help='Path to pretrained GAN state')
@click.option('--load/--no-load', default=False, show_default=True, help='Should an existing hyperopt run be loaded')
@click.option('--load-algo', type=click.Path(), help='Path to load hyperopt algorithm state from')
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR, show_default=True,
    help='Output directory for models and images'
)
def hyperopt(source_dir, only, target_imgs, target_annotations, eval_imgs, eval_annotations, masks, batch_size, dataloader_workers, epochs, samples, name, load_gan, load, load_algo, out_dir):
    '''
    Optimize DIHE hyperparameters.
    '''
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

@dihe.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to GP training image root'
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True,
    help='Path to GP test image root'
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True,
    help='Path to GP-180 annotation root'
)
@click.option('--model', type=click.Choice(('vgg16', 'resnet50')), default='vgg16', show_default=True, help='Base for the encoder model')
@click.option('--resnet-layers', type=int, multiple=True, default=[2, 3], show_default=True,
    help='If model is resnet50, which blocks should the descriptor be extracted from')
@click.option('--batch-norm/--no-batch-norm', default=False, show_default=True, help='Use/don\'t use batch normalization in encoder')
@click.option('--batch-size', type=int, default=8, show_default=True, help='Batch size')
@click.option('--dataloader-workers', type=int, default=8, show_default=True, help='Number of data loading processes')
@click.option('--enc-weights', help='Path to encoder weights - if not set, non-finetuned ImageNet weights are used')
@click.option('--only', type=click.Choice(('none', 'test', 'val')), default='none', show_default=True, help='Use all images (none) or only the test- or validation split')
@click.option('--knn', type=int, multiple=True, default=(1,), show_default=True, help='Consider classification correct if the correct class is among this many closest neighbours')
def eval(img_dir, test_imgs, annotations, model, resnet_layers, batch_norm, batch_size, dataloader_workers, enc_weights, only, knn):
    '''
    Evaluate classification performance.
    '''
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

@dihe.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to GP training image root'
)
@click.option(
    '--test-imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_TEST_DIR, show_default=True,
    help='Path to GP test image root'
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=GP_ANN_DIR, show_default=True,
    help='Path to GP-180 annotation root'
)
@click.option('--batch-norm/--no-batch-norm', default=False, show_default=True, help='Use/don\'t use batch normalization in encoder')
@click.option('--enc-weights', help='Path to encoder weights - if not set, non-finetuned ImageNet weights are used')
@click.option('--only', type=click.Choice(('none', 'test', 'val')), default='none', show_default=True, help='Use all images (none) or only the test- or validation split')
@click.option('--knn', type=int, default=4, show_default=True, help='How many nearest neighbours to show')
@click.option('--load-classifier-index', type=click.Path(), help='Load pre-embedded images from given path instead of calculating the embedding on the fly')
def visualize_performance(img_dir, test_imgs, annotations, batch_norm, enc_weights, only, knn, load_classifier_index):
    '''
    Visualize classification performance.

    Creates a plot with training --knn + 1 test images on the left column
    and --knn training images corresponding to the nearest classifications of each training image
    to the right of them.
    '''
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

@dihe.command()
@click.option(
    '--img-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    multiple=True,
    default=GP_TRAIN_FOLDERS, show_default=True,
    help='Path to training image root'
)
@click.option('--datatype', type=click.Choice(('gp', 'internal')), default='gp', show_default=True,
    help='Dataset type; gp for Grocery Product, internal for our internal dataset')
@click.option(
    '--out-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True),
    default=OUT_DIR, show_default=True,
    help='Output directory for embedded images'
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def prebuild_index(img_dir, datatype, out_dir, dihe_state):
    '''
    Pre-embed images.

    Passes all the images in the training set through the encoder
    and saves the resulting embedding vectors.
    The resulting file can be used with the --load-classifier-index -option
    in various commands.
    '''
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
