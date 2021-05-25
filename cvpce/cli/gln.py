import os
from functools import partial

import click
import torch
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from torch import multiprocessing as mp

from .. import datautils, proposals_training, proposals_eval, utils
from ..defaults import SKU110K_IMG_DIR, SKU110K_ANNOTATION_FILE, SKU110K_SKIP, OUT_DIR
from ..models import proposals

@click.group()
def gln():
    pass

@gln.command()
@click.option('--gln', type=click.Choice(['logging', 'plain']), default='logging')
@click.option(
    '--input-sizes', multiple=True, type = (int, int),
    default = [(300, 200), (200, 300), (600, 400)]
)
def build_assistant(gln, input_sizes):
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

@gln.command()
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
def train(imgs, annotations, eval_annotations, out_dir, method, tanh, batch_size, dataloader_workers, epochs, gpus, load, trim_module_prefix, hyperopt_params):
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

@gln.command()
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
def hyperopt(imgs, annotations, eval_annotations, name, batch_size, dataloader_workers, epochs, samples, load, load_algo, out_dir):
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

@gln.command()
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
def eval(dataset, imgs, annotations, batch_size, dataloader_workers, metric_workers, iou_threshold, coco, trim_module_prefix, plots, plot_res_reduction, state_file):
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

@gln.command()
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
