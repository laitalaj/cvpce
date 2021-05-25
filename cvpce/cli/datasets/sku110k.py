import random, re

import click
from torch.utils.data import DataLoader

from ... import datautils, proposals_eval, utils
from ...defaults import SKU110K_IMG_DIR, SKU110K_ANNOTATION_FILE

@click.group()
def sku110k():
    pass

@sku110k.command()
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
def visualize(imgs, annotations, index, method, flip, gaussians, model, conf_thresh, save, save_gaussians):
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

@sku110k.command()
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

@sku110k.command()
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
def iter(imgs, annotations):
    data = datautils.SKU110KDataset(imgs, annotations, include_gaussians=False)
    loader = DataLoader(data, batch_size=1, num_workers=8, collate_fn=datautils.sku110k_collate_fn)
    for i, d in enumerate(loader):
        if i % 100 == 0:
            print(i)
