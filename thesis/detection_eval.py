import torch
from torch.utils import data as tdata

from . import datautils, metrics, production

def evaluate_detections(p_model, c_model, testset, trainset, thresholds=(0.5,), proposal_batch_size=2, classification_batch_size=16, num_workers=8):
    test_loader = tdata.DataLoader(testset,
        batch_size=proposal_batch_size, num_workers=num_workers,
        collate_fn=datautils.sku110k_no_gauss_collate_fn, pin_memory=True,
    )
    classifier = production.Classifier(c_model, trainset,batch_size=classification_batch_size, num_workers=num_workers)

    predictions = {c: [] for c in range(len(testset.int_to_ann))}
    targets = {c: [] for c in range(len(testset.int_to_ann))}
    confidences = {c: [] for c in range(len(testset.int_to_ann))}

    print('Eval start!')
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0: print(f'{i}...')
            images, batch_targets = batch.cuda(non_blocking = True)
            result = p_model(images)

            for img, r, t in zip(images, result, batch_targets):
                to_classify = torch.stack([datautils.resize_for_classification(img[:, y1:y2, x1:x2]) for x1, y1, x2, y2 in r['boxes'].to(dtype=torch.long)])
                classes = classifier.classify(to_classify)
                class_tensor = torch.tensor([testset.ann_to_int[ann[0]] if ann[0] in testset.ann_to_int else -1 for ann in classes])
                class_set = {c.item() for c in class_tensor} | {l.item() for l in t['labels']}
                for c in class_set:
                    if c == -1: continue
                    predictions[c].append(r['boxes'][class_tensor == c].detach().cpu())
                    confidences[c].append(r['scores'][class_tensor == c].detach().cpu())
                    targets[c].append(t['boxes'][t['labels'] == c].detach().cpu())

    print('All data passed through model! Calculating metrics...')
    res = {c: metrics.calculate_metrics(targets[c], predictions[c], confidences[c], thresholds) for c in range(len(testset.int_to_ann))}
    print('Metrics calculated, eval done!')
    return {c: {thresh: {k: v for k, v in itm.items() if k != 'raw'} for thresh, itm in r.items()} for c, r in res.items()}

def mean_average_metrics(metrics, thresholds):
    return {t: {
        'map': sum(d[t]['ap'] for d in metrics.values()) / len(metrics),
        'mar300': sum(d[t]['ar_300'] for d in metrics.values()) / len(metrics)
        } for t in thresholds}

