import torch
from torchvision import ops as tvops

from . import datautils, production

def eval_dihe(encoder, sampleset, testset, batch_size, num_workers):
    print('Preparing classifier...')
    encoder.requires_grad_(False)

    classifier = production.Classifier(encoder, sampleset, batch_size=batch_size, num_workers=num_workers)

    total = 0
    correct = 0

    print('Eval start!')
    for i, (img, target_anns, boxes) in enumerate(testset):
        if i % 10 == 0:
            print(f'{i}...')

        boxes = tvops.clip_boxes_to_image(boxes, (img.shape[1], img.shape[2]))
        imgs = torch.stack([datautils.resize_for_classification(img[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes])
        pred_anns = classifier.classify(imgs)

        total += len(target_anns)
        for a1, a2 in zip(target_anns, pred_anns):
            if a1 == a2: correct += 1

    del classifier # maybe this will solve memory problems caused by eval?

    encoder.requires_grad_(True)

    print(f'Total annotations: {total}, Correctly guessed: {correct}, Accuracy: {correct / total:.4f}')
    return correct / total
