import torch
from torchvision import ops as tvops

from . import datautils, production

def eval_dihe(encoder, sampleset, testset, batch_size, num_workers, k=(1,), verbose=True):
    if verbose: print('Preparing classifier...')
    encoder.requires_grad_(False)

    classifier = production.Classifier(encoder, sampleset, batch_size=batch_size, num_workers=num_workers, k=max(k))

    total = 0
    correct = {knn: 0 for knn in k}
    missed = {}
    misclassification = {}
    total_per_ann = {}

    if verbose: print('Eval start!')
    for i, (img, target_anns, boxes) in enumerate(testset):
        if verbose and i % 10 == 0:
            print(f'{i}...')

        boxes = tvops.clip_boxes_to_image(boxes, (img.shape[1], img.shape[2]))
        imgs = torch.stack([datautils.resize_for_classification(img[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes])
        pred_anns = classifier.classify(imgs)

        total += len(target_anns)
        for a1, a2 in zip(target_anns, pred_anns):
            if a1 not in total_per_ann:
                total_per_ann[a1] = 0
            total_per_ann[a1] += 1

            for knn in k:
                if a1 in a2[:knn]: correct[knn] += 1
            if a1 != a2[0]:
                if a1 not in missed:
                    missed[a1] = 0
                    misclassification[a1] = {}
                if a2[0] not in misclassification[a1]:
                    misclassification[a1][a2[0]] = 0
                missed[a1] += 1
                misclassification[a1][a2[0]] += 1

    del classifier # maybe this will solve memory problems caused by eval?

    encoder.requires_grad_(True)

    accuracy = {knn: c / total for knn, c in correct.items()}
    if verbose:
        print(f'Total annotations: {total}, Correctly guessed: {correct}, Accuracy: {accuracy}')
        most_missed = sorted(((v / total_per_ann[k], v, k) for k, v in missed.items()), reverse=True)[:10]
        print(f'Most missed: {", ".join(f"{a} ({n}, {p * 100} %)" for p, n, a in most_missed)}')
        for _, n, k in most_missed[:3]:
            common_misclassifications = sorted(((v / n, v, k) for k, v in misclassification[k].items()), reverse=True)[:3]
            print(f'{k}: Commonly mistaken for {", ".join(f"{a} ({n}, {p * 100} %)" for p, n, a in common_misclassifications)}')
    return accuracy
