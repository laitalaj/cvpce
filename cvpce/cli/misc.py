import click
from pycocotools import cocoeval
from torchvision import datasets as dsets
from torchvision import models as tmodels
from torchvision import transforms as tforms

from .. import utils
from ..defaults import COCO_IMG_DIR, COCO_ANNOTATION_FILE

@click.group()
def misc():
    '''
    Miscellancellous commands.

    Currently, this command group contains only a command for testing the pre-trained
    PyTorch RetinaNet against COCO
    that I used early on when building this to make sure that the pre-trained weights are up to the task.
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
