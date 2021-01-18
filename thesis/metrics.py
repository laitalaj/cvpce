import heapq

import torch
from torchvision import ops as tvops
from matplotlib import pyplot as plt

# The precision at each recall level r is interpolated by taking the maximum precision measured for a method for which the corresponding recall exceeds r
# assigned to groundtruth objects satisfying the overlap criterion in order ranked by the (decreasing) confidence output

def iou_matrices(targets, sorted_predictions):
    ious = tvops.box_iou(sorted_predictions, targets)
    return torch.sort(ious, dim=1, descending=True)

def check_matches(sorted_ious, indices, iou_threshold=0.5):
    predictions, targets = sorted_ious.shape

    used = torch.zeros(targets)
    true_positive = torch.zeros(predictions)
    false_positive = torch.zeros(predictions)
    for i, (single_ious, single_idxs) in enumerate(zip(sorted_ious, indices)):
        match = False
        for iou, idx in zip(single_ious, single_idxs):
            if iou < iou_threshold: break
            if used[idx]: continue
            used[idx] = 1
            match = True
        if match:
            true_positive[i] = 1
        else:
            false_positive[i] = 1

    return true_positive, false_positive

def merge_matches(true_positives, false_positives, confidences, total_predictions=None): # assuming confidences sorted, e.g. matching t&fp order, here
    def to_heap_element(idx, conf, tp, fp):
        return (-conf[idx], idx, conf, tp, fp)

    if total_predictions is None:
        total_predictions = sum(len(t) for t in true_positives)

    merged_tp = torch.zeros(total_predictions)
    merged_fp = torch.zeros(total_predictions)

    heap = [to_heap_element(0, conf, tp, fp) for conf, tp, fp in zip(confidences, true_positives, false_positives)]
    heapq.heapify(heap)

    for i in range(total_predictions):
        _, idx, conf, tp, fp = heap[0]
        merged_tp[i] = tp[idx]
        merged_fp[i] = fp[idx]
        idx += 1
        if idx < len(conf):
            heapq.heapreplace(heap, to_heap_element(idx, conf, tp, fp))
        else:
            heapq.heappop(heap)

    assert len(heap) == 0, 'All lists were not fully merged, this should not happen! Was total_predictions correct?'

    return merged_tp, merged_fp

def precision_and_recall(true_positives, false_positives, total_targets):
    true_positives = true_positives.cumsum(0)
    false_positives = false_positives.cumsum(0)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / total_targets
    return precision, recall

def f_score(precision, recall):
    res = 2 * precision * recall / (precision + recall)
    res[torch.isnan(res)] = 0
    return res

def average_precision(precision, recall):
    values = torch.zeros(11, dtype=torch.float)
    for i, r in enumerate(torch.linspace(0, 1, 11)):
        precision_at_recall = precision[recall >= r]
        if len(precision_at_recall) > 0:
            values[i] = precision_at_recall.max()
        else: break # if there were no precisions for recall r1, there won't be any for recall r2 > r1
    return values.mean()

def calculate_metrics(targets, predictions, confidences, iou_thresholds = (0.5,)):
    matches_for_threshold = {t: {'true_positives': [], 'false_positives': []} for t in iou_thresholds}
    sorted_confidences = []
    total_predictions = 0
    total_targets = 0
    for target, prediction, confidence in zip(targets, predictions, confidences):
        confidence, sort_idx = torch.sort(confidence)
        prediction = prediction[sort_idx]

        sorted_confidences.append(confidence)
        total_predictions += prediction.shape[0]
        total_targets += target.shape[0]

        iou_matrix, index_matrix = iou_matrices(target, prediction)
        for t in iou_thresholds:
            tp, fp = check_matches(iou_matrix, index_matrix, t)
            matches_for_threshold[t]['true_positives'].append(tp)
            matches_for_threshold[t]['false_positives'].append(fp)

    res = {}
    for t in iou_thresholds:
        tp, fp = merge_matches(
            matches_for_threshold[t]['true_positives'],
            matches_for_threshold[t]['false_positives'],
            sorted_confidences,
            total_predictions,
        )
        p, r = precision_and_recall(tp, fp, total_targets)
        f = f_score(p, r)
        max_f, max_idx = f.max(0)
        res[t] = {
            'raw': {
                'p': p,
                'r': r,
                'f': f
            },
            'f': max_f,
            'p': p[max_idx],
            'r': r[max_idx],
            'ap': average_precision(p, r),
        }
    return res

def plot_prf(precision, recall, fscore):
    plt.plot(recall, precision, label='Precision')
    plt.plot(recall, fscore, label='$F_1$')
    plt.vlines(recall[fscore.argmax()], 0, 1, color='red', label='Max $F_1$')
    plt.xlabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
