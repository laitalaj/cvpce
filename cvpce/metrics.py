import heapq
import multiprocessing as mp

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

def merge_matches(matches, confidences): # NOT assuming confidences sorted
    merged_conf = torch.cat(confidences)
    merged_conf, sort_idx = torch.sort(merged_conf, descending=True)

    merged_matches = {t: {
            'true_positives': torch.cat(d['true_positives'])[sort_idx],
            'false_positives': torch.cat(d['false_positives'])[sort_idx],
            'ar_300': sum(d['recall_300']) / len(d['recall_300']),
        } for t, d in matches.items()}

    return merged_matches, merged_conf

def get_merge_index(c1, c2):
    return torch.sort(torch.cat((c1, c2)), descending=True)

def precision_and_recall(true_positives, false_positives, total_targets):
    true_positives = true_positives.cumsum(0)
    false_positives = false_positives.cumsum(0)

    precision = true_positives / (true_positives + false_positives)
    precision[torch.isnan(precision)] = 0

    recall = true_positives / total_targets if total_targets > 0 else torch.zeros_like(true_positives)

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

def _process_one(target, prediction, confidence, iou_thresholds):
    confidence, sort_idx = torch.sort(confidence, descending=True)
    prediction = prediction[sort_idx]

    iou_matrix, index_matrix = iou_matrices(target, prediction)
    matches_for_threshold = {}
    for t in iou_thresholds:
        tp, fp = check_matches(iou_matrix, index_matrix, t)
        _, r = precision_and_recall(tp, fp, len(target))
        matches_for_threshold[t] = {
            'true_positives': tp,
            'false_positives': fp,
            'recall_300': r[:300][-1] if len(r) > 0 else 0,
        }

    return matches_for_threshold, confidence, target.shape[0]

def _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_targets):
    res = {}

    matches_for_threshold, conf = merge_matches(matches_for_threshold, sorted_confidences)

    for t in iou_thresholds:
        tp = matches_for_threshold[t]['true_positives']
        fp = matches_for_threshold[t]['false_positives']
        p, r = precision_and_recall(tp, fp, total_targets)
        f = f_score(p, r)
        if len(f) > 0:
            max_f, max_idx = f.max(0)
            best_p = p[max_idx]
            best_r = r[max_idx]
            conf_thresh = conf[max_idx]
        else:
            max_f, best_p, best_r, conf_thresh = 0.0, 0.0, 0.0, 0.0
        res[t] = {
            'raw': {
                'p': p,
                'r': r,
                'f': f,
                'c': conf,
            },
            'f': max_f,
            'p': best_p,
            'r': best_r,
            'c': conf_thresh,
            'ap': average_precision(p, r),
            'ar_300': matches_for_threshold[t]['ar_300'],
        }
    return res

def calculate_metrics(targets, predictions, confidences, iou_thresholds = (0.5,)):
    matches_for_threshold = {t: {'true_positives': [], 'false_positives': [], 'recall_300': []} for t in iou_thresholds}
    sorted_confidences = []
    total_targets = 0
    for target, prediction, confidence in zip(targets, predictions, confidences):
        matches, conf, targets = _process_one(target, prediction, confidence, iou_thresholds)
        sorted_confidences.append(conf)
        total_targets += targets
        for t in iou_thresholds:
            matches_for_threshold[t]['true_positives'].append(matches[t]['true_positives'])
            matches_for_threshold[t]['false_positives'].append(matches[t]['false_positives'])
            matches_for_threshold[t]['recall_300'].append(matches[t]['recall_300'])

    return _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_targets)

def _image_processer(input_queue, output_queue, iou_thresholds):
    for target, prediction, confidence in iter(input_queue.get, None):
        result = _process_one(target, prediction, confidence, iou_thresholds)
        output_queue.put(result)
        input_queue.task_done()
    input_queue.task_done()

def _metric_calculator(output_queue, pipe, iou_thresholds):
    matches_for_threshold = {t: {'true_positives': [], 'false_positives': [], 'recall_300': []} for t in iou_thresholds}
    sorted_confidences = []
    total_targets = 0
    for matches, conf, targets in iter(output_queue.get, None):
        sorted_confidences.append(conf)
        total_targets += targets
        for t in iou_thresholds:
            matches_for_threshold[t]['true_positives'].append(matches[t]['true_positives'])
            matches_for_threshold[t]['false_positives'].append(matches[t]['false_positives'])
            matches_for_threshold[t]['recall_300'].append(matches[t]['recall_300'])
        output_queue.task_done()

    res = _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_targets)
    pipe.send(res)
    output_queue.task_done()
    print(f'Output queue is empty: {output_queue.empty()}')

def calculate_metrics_async(processes = 4, iou_thresholds = (0.5,)):
    input_queue = mp.JoinableQueue()
    output_queue = mp.JoinableQueue()
    out_pipe, in_pipe = mp.Pipe()

    for _ in range(processes):
        mp.Process(target=_image_processer, args=(input_queue, output_queue, iou_thresholds)).start()

    mp.Process(target=_metric_calculator, args=(output_queue, in_pipe, iou_thresholds)).start()

    return input_queue, output_queue, out_pipe

def plot_prfc(precision, recall, fscore, confidence, title=None, resolution_reduction=1):
    fig = plt.figure(figsize=(5, 2.5))

    f_max_idx = fscore.argmax()
    plt.vlines(recall[f_max_idx], 0, 1, color='red', label='Max. $F_1$')
    plt.hlines(confidence[f_max_idx], 0, recall[f_max_idx], color='orange', linestyles='dashed')
    plt.hlines(precision[f_max_idx], 0, recall[f_max_idx], color='blue', linestyles='dashed')
    plt.hlines(fscore[f_max_idx], 0, recall[f_max_idx], color='green', linestyles='dashed')

    plt.annotate(f'{recall[f_max_idx]:.2f}', (recall[f_max_idx], 0), annotation_clip=False, color='red', ha='center', va='top')
    plt.annotate(f'{confidence[f_max_idx]:.2f}', (0, confidence[f_max_idx]), annotation_clip=False, color='orange', ha='right', va='center')
    plt.annotate(f'{precision[f_max_idx]:.2f}', (0, precision[f_max_idx]), annotation_clip=False, color='blue', ha='right', va='center')
    plt.annotate(f'{fscore[f_max_idx]:.2f}', (0, fscore[f_max_idx]), annotation_clip=False, color='green', ha='right', va='center')

    plt.plot(recall[::resolution_reduction], confidence[::resolution_reduction], label='Confidence', color='orange')
    plt.plot(recall[::resolution_reduction], precision[::resolution_reduction], label='Precision', color='blue')
    plt.plot(recall[::resolution_reduction], fscore[::resolution_reduction], label='$F_1$', color='green')

    if title is not None:
        plt.title(title)
    plt.xlabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    fig.tight_layout(pad=0.5)

    plt.show()
