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

def merge_matches(true_positives, false_positives, confidences, total_predictions=None): # assuming confidences sorted, e.g. matching t&fp order, here
    def to_heap_element(elem_id, idx, conf, tp, fp):
        return (-conf[idx], elem_id, idx, conf, tp, fp) # elem_id included to break confidence ties

    if total_predictions is None:
        total_predictions = sum(len(t) for t in true_positives)

    merged_tp = torch.zeros(total_predictions)
    merged_fp = torch.zeros(total_predictions)
    merged_conf = torch.zeros(total_predictions, dtype=torch.float)

    heap = [to_heap_element(i, 0, conf, tp, fp) for i, (conf, tp, fp) in enumerate(zip(confidences, true_positives, false_positives)) if len(conf) > 0]
    heapq.heapify(heap)

    for i in range(total_predictions):
        _, elem_id, idx, conf, tp, fp = heap[0]
        merged_tp[i] = tp[idx]
        merged_fp[i] = fp[idx]
        merged_conf[i] = conf[idx]
        idx += 1
        if idx < len(conf):
            heapq.heapreplace(heap, to_heap_element(elem_id, idx, conf, tp, fp))
        else:
            heapq.heappop(heap)

    assert len(heap) == 0, 'All lists were not fully merged, this should not happen! Was total_predictions correct?'

    return merged_tp, merged_fp, merged_conf

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
        matches_for_threshold[t] = {
            'true_positives': tp,
            'false_positives': fp,
        }

    return matches_for_threshold, confidence, prediction.shape[0], target.shape[0]

def _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_predictions, total_targets):
    res = {}
    for t in iou_thresholds:
        tp, fp, conf = matches_for_threshold[t]['true_positives'], matches_for_threshold[t]['false_positives'], sorted_confidences

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
        }
    return res

def calculate_metrics(targets, predictions, confidences, iou_thresholds = (0.5,)):
    matches_for_threshold = {t: {'true_positives': torch.empty(0, dtype=torch.long), 'false_positives': torch.empty(0, dtype=torch.long)} for t in iou_thresholds}
    sorted_confidences = torch.empty(0, dtype=torch.float)
    total_predictions = 0
    total_targets = 0
    for target, prediction, confidence in zip(targets, predictions, confidences):
        matches, conf, predictions, targets = _process_one(target, prediction, confidence, iou_thresholds)
        sorted_confidences, merge_idx = get_merge_index(sorted_confidences, conf)
        total_predictions += predictions
        total_targets += targets
        for t in iou_thresholds:
            matches_for_threshold[t]['true_positives'] = torch.cat((matches_for_threshold[t]['true_positives'], matches[t]['true_positives']))[merge_idx]
            matches_for_threshold[t]['false_positives'] = torch.cat((matches_for_threshold[t]['false_positives'], matches[t]['false_positives']))[merge_idx]

    return _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_predictions, total_targets)

def _image_processer(input_queue, output_queue, iou_thresholds):
    for target, prediction, confidence in iter(input_queue.get, None):
        result = _process_one(target, prediction, confidence, iou_thresholds)
        output_queue.put(result)
        input_queue.task_done()
    input_queue.task_done()

def _metric_calculator(output_queue, pipe, iou_thresholds):
    matches_for_threshold = {t: {'true_positives': torch.empty(0, dtype=torch.long), 'false_positives': torch.empty(0, dtype=torch.long)} for t in iou_thresholds}
    sorted_confidences = torch.empty(0, dtype=torch.float)
    total_predictions = 0
    total_targets = 0
    for i, (matches, conf, predictions, targets) in enumerate(iter(output_queue.get, None)):
        if i % 100 == 0:
                print(f'Metric calculator: {i}...')
        sorted_confidences, merge_idx = get_merge_index(sorted_confidences, conf)
        total_predictions += predictions
        total_targets += targets
        for t in iou_thresholds:
            matches_for_threshold[t]['true_positives'] = torch.cat((matches_for_threshold[t]['true_positives'], matches[t]['true_positives']))[merge_idx]
            matches_for_threshold[t]['false_positives'] = torch.cat((matches_for_threshold[t]['false_positives'], matches[t]['false_positives']))[merge_idx]
        output_queue.task_done()

    print('-> Calculating the actual metrics...')
    res = _do_calculate(iou_thresholds, matches_for_threshold, sorted_confidences, total_predictions, total_targets)
    pipe.send(res)
    output_queue.task_done()
    print(f'Done! Output queue is empty: {output_queue.empty()}')

def calculate_metrics_async(processes = 4, iou_thresholds = (0.5,)):
    input_queue = mp.JoinableQueue()
    output_queue = mp.JoinableQueue()
    out_pipe, in_pipe = mp.Pipe()

    for _ in range(processes):
        mp.Process(target=_image_processer, args=(input_queue, output_queue, iou_thresholds)).start()

    mp.Process(target=_metric_calculator, args=(output_queue, in_pipe, iou_thresholds)).start()

    return input_queue, output_queue, out_pipe

def plot_prfc(precision, recall, fscore, confidence):
    plt.plot(recall, confidence, label='Confidence')
    plt.plot(recall, precision, label='Precision')
    plt.plot(recall, fscore, label='$F_1$')
    plt.vlines(recall[fscore.argmax()], 0, 1, color='red', label='Max $F_1$')
    plt.xlabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
