import torch

from thesis import metrics

TARGETS = [
    torch.tensor([[0, 0, 1, 1], [1, 0, 2, 1], [1, 1, 2, 2]], dtype=torch.float),
    torch.tensor([[1, 1, 2, 2], [3, 1, 4, 2], [5, 1, 6, 2], [7, 1, 8, 2]], dtype=torch.float),
    torch.tensor([[0, 0, 5, 5], [5, 5, 10, 10]], dtype=torch.float),
]

PREDICTIONS = [
    torch.tensor([[0, 0, .9, .9], [1.1, 0.1, 1.9, 0.9], [0, 0, 1, 1], [0.9, 0.9, 2.1, 2.1], [3, 3, 4, 4]], dtype=torch.float), # IoU: 0.81, 0.64, 1.0 (but duplicate of first), 0.69, 0
    torch.tensor([[1, 0, 2, 1], [1, 1, 2, 2], [5, 1, 6, 2], [7, 1.1, 8, 1.9], [9, 9, 10, 10]], dtype=torch.float), # IoU: 0, 1, 1, 0.8, 0
    torch.tensor([[0, 0, 1, 1], [1, 1, 3, 3], [0.5, 0.5, 4.5, 4.5], [0, 0, 6, 6], [6, 6, 9, 9]], dtype=torch.float), # IoU: 0.04, 0.16 (duplicate), 0.64 (duplicate), 0.69 (duplicate), 0.36
]

CONFIDENCES = [
    torch.tensor([1, 0.8, 0.6, 0.4, 0.2], dtype=torch.float),
    torch.tensor([0.9, 0.8, 0.7, 0.65, 0.5], dtype=torch.float),
    torch.tensor([0.85, 0.6, 0.4, 0.2, 0.1], dtype=torch.float),
]

def test_iou_matrices():
    expected_ious = torch.tensor([
        [0.04, 0],
        [0.16, 0],
        [0.64, 0],
        [(5*5) / (6*6), 1 / (5*5 + 6*6 - 1)],
        [0.36, 0]
    ], dtype=torch.float)
    expected_indices = torch.tensor([
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [1, 0]
    ])
    ious, indices = metrics.iou_matrices(TARGETS[2], PREDICTIONS[2])
    assert expected_indices.equal(indices)
    assert expected_ious.allclose(ious)

def test_iou_matrices_2():
    expected_ious = torch.tensor([
        [0.81, 0, 0],
        [0.64, 0, 0],
        [1, 0, 0],
        [1/1.44, 0.1 / (1.44 + 1 - 0.1), 0.01 / (1.44 + 1 - 0.01)],
        [0, 0, 0]
    ], dtype=torch.float)
    expected_indices = torch.tensor([
        [0, 1, 2],
        [1, 0, 2],
        [0, 1, 2],
        [2, 1, 0],
        [0, 1, 2]
    ])
    ious, indices = metrics.iou_matrices(TARGETS[0], PREDICTIONS[0])
    assert expected_indices.equal(indices)
    assert expected_ious.allclose(ious)

def test_check_matches():
    expected_tp = torch.tensor([1, 0, 0, 1, 0], dtype=torch.float)
    expected_fp = torch.ones_like(expected_tp) - expected_tp
    
    ious, indices = metrics.iou_matrices(TARGETS[0], PREDICTIONS[0])
    tp, fp = metrics.check_matches(ious, indices, iou_threshold=0.65)
    assert expected_tp.allclose(tp) # TODO: consider using some other tensor type for TP&FP
    assert expected_fp.allclose(fp)

def tps_fps():
    tps = []
    fps = []
    for target, prediction in zip(TARGETS, PREDICTIONS):
        ious, indices = metrics.iou_matrices(target, prediction)
        tp, fp = metrics.check_matches(ious, indices)
        tps.append(tp)
        fps.append(fp)
    return {0.5: {'true_positives': tps, 'false_positives': fps}}

def unpack_matches(matches):
    assert len(matches) == 1
    assert 0.5 in matches
    assert len(matches[0.5]) == 2
    assert 'true_positives' in matches[0.5] and 'false_positives' in matches[0.5]
    return matches[0.5]['true_positives'], matches[0.5]['false_positives']

def test_merge_matches():
    expected_tp = torch.tensor([1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.float)
    expected_fp = torch.ones_like(expected_tp) - expected_tp
    expected_conf = torch.tensor([1, 0.9, 0.85, 0.8, 0.8, 0.7, 0.65, 0.6, 0.6, 0.5, 0.4, 0.4, 0.2, 0.2, 0.1], dtype=torch.float)

    matches = tps_fps()
    matches, conf = metrics.merge_matches(matches, CONFIDENCES)
    tp, fp = unpack_matches(matches)
    assert expected_tp.allclose(tp)
    assert expected_fp.allclose(fp)
    assert expected_conf.allclose(conf)

def test_precision_recall():
    expected_precision = torch.tensor([1, 1/2, 1/3, 2/4, 3/5, 4/6, 5/7, 5/8, 5/9, 5/10, 6/11, 7/12, 7/13, 7/14, 7/15])
    expected_recall =  torch.tensor([1/9, 1/9, 1/9, 2/9, 3/9, 4/9, 5/9, 5/9, 5/9, 5/9,  6/9,  7/9,  7/9,  7/9,  7/9])

    matches = tps_fps()
    matches, _ = metrics.merge_matches(matches, CONFIDENCES)
    tp, fp = unpack_matches(matches)
    p, r = metrics.precision_and_recall(tp, fp, sum(len(t) for t in TARGETS))
    assert expected_precision.allclose(p)
    assert expected_recall.allclose(r)

def test_ap():
    expected_ap = torch.tensor((1 + 1 + 5/7 + 5/7 + 5/7 + 5/7 + 7/12 + 7/12 + 0 + 0 + 0) / 11)

    matches = tps_fps()
    matches, _ = metrics.merge_matches(matches, CONFIDENCES)
    tp, fp = unpack_matches(matches)
    p, r = metrics.precision_and_recall(tp, fp, sum(len(t) for t in TARGETS))
    ap = metrics.average_precision(p, r)
    assert expected_ap.isclose(ap)

def test_calculate_metrics():
    expected_precision = torch.tensor(7/12)
    expected_recall =  torch.tensor(7/9)
    expected_f = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    expected_ap = torch.tensor((1 + 1 + 5/7 + 5/7 + 5/7 + 5/7 + 7/12 + 7/12 + 0 + 0 + 0) / 11)

    res = metrics.calculate_metrics(TARGETS, PREDICTIONS, CONFIDENCES)
    assert torch.isclose(res[0.5]['ap'], expected_ap)
    assert torch.isclose(res[0.5]['p'], expected_precision)
    assert torch.isclose(res[0.5]['r'], expected_recall)
    assert torch.isclose(res[0.5]['f'], expected_f)
