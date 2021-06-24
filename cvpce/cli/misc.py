import json

import click
import PIL as pil
from pycocotools import cocoeval
import torch
from torchvision import datasets as dsets
from torchvision import models as tmodels
from torchvision import transforms as tforms
import torchvision.ops as tvops
import torchvision.transforms.functional as ttf
import matplotlib.pyplot as plt

from .. import planograms, utils
from ..datautils import SimpleFolderSet, resize_for_classification
from ..defaults import COCO_IMG_DIR, COCO_ANNOTATION_FILE
from ..models import proposals, classification
from ..production import ProposalGenerator, Classifier
from ..proposals_training import MODEL_STATE_DICT_KEY
from ..classification_training import EMBEDDER_STATE_DICT_KEY

@click.group()
def misc():
    '''
    Miscellancellous commands.
    '''
    pass

@misc.command()
@click.option(
    '--imgs',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=COCO_IMG_DIR, show_default=True,
    help='Path to COCO image directory',
)
@click.option(
    '--annotations',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default=COCO_ANNOTATION_FILE, show_default=True,
    help='Path to COCO annotation file',
)
@click.option(
    '--score-threshold',
    type=float,
    default=0.05, show_default=True,
    help='Minimum prediction score to keep for evaluation',
)
@click.option(
    '--limit',
    type=int,
    default=-1, show_default=True,
    help='Process at most this many images, -1 to process all',
)
def retinanet_coco_test(imgs, annotations, score_threshold, limit):
    '''
    Test pre-trained PyTorch RetinaNet against COCO.

    Loads up a RetinaNet, runs data through it according to the provided annotation file
    and evaluates the results using pycocotools COCOeval.
    I used this to make sure that the PyTorch-provided RetinaNet works fine
    and to familiarize myself with it!
    '''
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


@misc.command()
@click.argument('gln-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dihe-state',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('dataset-folder',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True)
)
@click.argument('image-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
@click.argument('plano-file',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True)
)
def pipeline_demo(gln_state, dihe_state, dataset_folder, image_file, plano_file):
    '''
    Demonstrate the CVPCE pipeline.

    The dataset folder is expected to contain .png, .jpg and .jpeg files,
    one for each class, with the class label set as the filename.

    The planogram file should be a JSON file formatted in a manner such as

    \b
    [
        {
            "label": "class1",
            "box": [0, 0, 5, 5]
        },
        {
            "label": "class2",
            "box": [5, 0, 10, 5]
        }
    ]

    (in this case, the dataset folder should contain files for class1 and class2,
    such as class1.png and class2.jpg)
    '''
    # TODO: This shares a bunch of code with cvpce plot-planogram-eval; will want to refactor a bit at some point

    def double_fig(img):
        return plt.subplots(1, 2, figsize=(12, 12)) if img.shape[2] < img.shape[1] else plt.subplots(2, 1, figsize=(12, 12))

    dataset = SimpleFolderSet(dataset_folder)
    rebuildset = SimpleFolderSet(dataset_folder, train=False)

    state_dict = torch.load(gln_state)[MODEL_STATE_DICT_KEY]
    gln = proposals.gln().cuda()
    gln.load_state_dict(state_dict)
    gln.eval()
    gln.requires_grad_(False)
    generator = ProposalGenerator(gln)

    img = ttf.to_tensor(pil.Image.open(image_file))
    detections, images = generator.generate_proposals_and_images(img)

    encoder = classification.macvgg_embedder(model='vgg16', pretrained=False).cuda()
    enc_state = torch.load(dihe_state)
    encoder.load_state_dict(enc_state[EMBEDDER_STATE_DICT_KEY])
    encoder.eval()
    encoder.requires_grad_(False)
    classifier = Classifier(encoder, dataset)

    classes, embedding = classifier.classify(images, return_embedding=True)

    with open(plano_file) as pf:
        plano = json.load(pf)
    expected_boxes = torch.tensor([o['box'] for o in plano], dtype=torch.float)
    expected_labels = [o['label'] for o in plano]
    actual_boxes = detections.detach().cpu()
    actual_labels = [c[0] for c in classes]
    ge = planograms.build_graph(expected_boxes, expected_labels, thresh_size=0.7)
    ga = planograms.build_graph(actual_boxes, actual_labels, thresh_size=0.7)

    matching = planograms.large_common_subgraph(ge, ga)
    nodes_e, nodes_a = (list(l) for l in zip(*matching)) if len(matching) else ([],[])
    sge = ge.subgraph(nodes_e)
    sga = ga.subgraph(nodes_a)

    h, w = img.shape[1:]
    reproj_threshold = min(h, w) * 0.01

    _, found_actual, expected_positions, missing_indices, missing_positions, missing_labels = planograms.finalize_via_ransac(
        matching, expected_boxes, actual_boxes, expected_labels, actual_labels,
        reproj_threshold=reproj_threshold, return_matched_actual=True, return_expected_positions=True
    )
    missing_positions = tvops.clip_boxes_to_image(missing_positions, img.shape[1:])
    valid_positions = (missing_positions[:,2] - missing_positions[:,0] > 1) & (missing_positions[:,3] - missing_positions[:,1] > 1)
    missing_indices = missing_indices[valid_positions]
    missing_positions = missing_positions[valid_positions]
    missing_labels = [l for l, v in zip(missing_labels, valid_positions) if v]

    if len(missing_positions) > 0:
        found_round2 = torch.full((len(missing_indices),), False)
        missing_imgs = torch.stack([resize_for_classification(img[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in missing_positions.to(dtype=torch.long)])
        reclass_labels = classifier.classify(missing_imgs)
        for idx, (expected_label, actual_label) in enumerate(zip(missing_labels, reclass_labels)):
            if expected_label == actual_label[0]:
                found_round2[idx] = True

    _, (ax1, ax2) = double_fig(img)
    utils.build_fig(img, ax=ax1)
    utils.build_rebuild(expected_boxes, expected_labels, rebuildset, ax=ax2)
    ax1.set_title('Image to evaluate')
    ax2.set_title('Planogram')
    plt.show()

    utils.show(img, utils.recall_tensor(tvops.box_convert(detections, 'xyxy', 'xywh')))

    data_imgs = torch.stack([i for i, _, _, _ in dataset])
    utils.show_demo_emb_fig(data_imgs, classifier.embedding, images, embedding, draw_positives=False)

    utils.show_demo_emb_fig(data_imgs, classifier.embedding, images, embedding)

    _, (ax1, ax2) = double_fig(img)
    utils.build_fig(img, ax=ax1)
    utils.build_rebuild(detections, actual_labels, rebuildset, ax=ax2)
    ax1.set_title('Image')
    ax2.set_title('Classified detections = "Observed planogram"')
    plt.show()

    _, (ax1, ax2) = double_fig(img)
    utils.build_rebuild(expected_boxes, expected_labels, rebuildset, ax=ax1)
    utils.draw_planograph(ge, expected_boxes, ax=ax1, flip_y=True)
    utils.build_rebuild(actual_boxes, actual_labels, rebuildset, ax=ax2)
    utils.draw_planograph(ga, actual_boxes, ax=ax2, flip_y=True)
    ax1.set_title('Expected planogram')
    ax2.set_title('Observed planogram')
    plt.show()

    _, (ax1, ax2) = double_fig(img)
    utils.build_rebuild(expected_boxes, expected_labels, rebuildset, ax=ax1)
    utils.draw_planograph(sge, expected_boxes, ax=ax1, flip_y=True)
    utils.build_rebuild(actual_boxes, actual_labels, rebuildset, ax=ax2)
    utils.draw_planograph(sga, actual_boxes, ax=ax2, flip_y=True)
    ax1.set_title('Expected planogram')
    ax2.set_title('Observed planogram')
    plt.show()

    utils.show(img, tvops.box_convert(expected_positions, 'xyxy', 'xywh'))

    utils.build_fig(img,
        groundtruth=tvops.box_convert(actual_boxes[found_actual], 'xyxy', 'xywh'),
        detections=tvops.box_convert(missing_positions, 'xyxy', 'xywh'),
    )
    if len(missing_positions) > 0:
        utils.plot_boxes(tvops.box_convert(missing_positions[found_round2], 'xyxy', 'xywh'), color='yellow', hl_color='orange')
    plt.show()
