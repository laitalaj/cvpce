import random

import click
import torch
from matplotlib import pyplot as plt
from torchvision import ops as tvops

from .. import classification_training, datautils, detection_eval, planograms, production, proposals_eval, utils
from ..defaults import GP_TRAIN_FOLDERS, GP_TEST_DIR, GP_ANN_DIR, GP_PLANO_DIR
from ..models import classification

@click.command()
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

@click.command()
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

@click.command()
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

@click.command()
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
    boxes = boxes.detach().cpu()
    classes = [ann[0] for ann in classifier.classify(images)]
    actual = {'boxes': boxes, 'labels': classes}

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
