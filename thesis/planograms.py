from math import pi

import torch
from torchvision import ops as tvops
import networkx as nx
from cv2 import findHomography, RANSAC

CARDINALS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'] # TODO: Could use integers instead of strings for this to gain some speed

def _check_dir(i, j, dir, matrices, graph, dist):
    if not matrices[dir][i,j]: # j not in direction dir from i
        return False

    dir_idx = CARDINALS.index(dir)
    opposite_dir = CARDINALS[(dir_idx + 4) % 8]
    for k in graph[j]:
        existing_edge = graph[j][k]
        if existing_edge['dir'] == opposite_dir:
            if existing_edge['weight'] <= dist:
                return False # A shorter edge already exists
            graph.remove_edge(j, k)
            graph.remove_edge(k, j)
            break
    graph.add_edge(i, j, dir=dir, weight=dist)
    graph.add_edge(j, i, dir=opposite_dir, weight=dist)
    return True

def build_graph(boxes, labels, thresh_size=0.5):
    minx = torch.amin(boxes[:,0])
    miny = torch.amin(boxes[:,1])
    maxx = torch.amax(boxes[:,2])
    maxy = torch.amax(boxes[:,3])
    avg_dim = (maxx - minx + maxy - miny) / 2
    thresh = thresh_size * avg_dim

    centres = torch.tensor([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes])
    dists = torch.cdist(centres[None], centres[None])[0]

    idxrange = torch.arange(len(centres))
    aa, bb = torch.meshgrid(idxrange, idxrange)
    dir_vecs = (centres[bb] - centres[aa]) / dists.reshape(len(centres), len(centres), 1) # Matrix of normalized direction vectors from aa to bb
    dirs = torch.acos(dir_vecs[:,:,0].clamp(-1, 1)) # Matrix of angles, clamping necessary due to float inaccuracy
    over_180 = dir_vecs[:,:,1] < 0 # y-component < 0 --> true angle is 360 - acos(x)
    dirs[over_180] = 2 * pi - dirs[over_180]

    dir_matrices = {
        'E': (dirs > 15 * pi / 8) | (dirs <= pi / 8),
        **{d: (dirs > (1 + 2 * i) * pi / 8) & (dirs <= (1 + 2 * (i + 1)) * pi / 8) for i, d in enumerate(CARDINALS[1:])}
    }

    g = nx.DiGraph()
    g.add_nodes_from([(i.item(), {'label': labels[i]}) for i in idxrange])
    sorted_dist, sort_idx = dists.sort(dim=1)
    for i in idxrange:
        i = i.item()
        not_found = set(CARDINALS)
        for neigh in g[i]:
            not_found.remove(g[i][neigh]['dir'])
        for d, j in zip(sorted_dist[i], sort_idx[i]):
            if d > thresh or not len(not_found): # Iterate until all cardinal dirs have been found or until we are over the dist threshold
                break
            j = j.item()
            if i == j: continue # No self-loops allowed
            for dir in not_found:
                if _check_dir(i, j, dir, dir_matrices, g, d):
                    not_found.remove(dir)
                    break

    return g, centres

def _build_hypothesis(g1, g2, n1, n2, edge_label):
    neigh1 = {}
    neigh2 = {}
    for nn1 in g1[n1]:
        label = g1[n1][nn1][edge_label]
        neigh1[label] = g1.nodes[nn1]
    for nn2 in g2[n2]:
        label = g2[n2][nn2][edge_label]
        neigh2[label] = g2.nodes[nn2]
    score = sum(neigh1[lbl] == neigh2[lbl] for lbl in neigh1 if lbl in neigh2)
    score /= len(CARDINALS) # alternatives: max_labels in example, len(neigh1) in paper, len(neigh1 | neigh2) is one possibility
    return (-score, n1, n2)

def build_hypotheses(g1, g2, edge_label = 'dir'):
    hypotheses = []
    for n1 in g1:
        for n2 in g2:
            if g1.nodes[n1] == g2.nodes[n2]:
                hypotheses.append(_build_hypothesis(g1, g2, n1, n2, edge_label))
 
    return sorted(hypotheses)

def _get_next(g1, g2, n1, n2, edge_label):
    nxt = []
    for e1 in g1[n1]:
        for e2 in g2[n2]:
            ed1 = g1[n1][e1][edge_label]
            ed2 = g2[n2][e2][edge_label]
            nd1 = g1.nodes[e1]
            nd2 = g2.nodes[e2]
            if ed1 == ed2 and nd1 == nd2:
                nxt.append((e1, e2))
    return nxt

def large_common_subgraph(g1, g2, edge_label = 'dir', min_score = -0.2, stop_at_fraction = 1/3):
    hypotheses = build_hypotheses(g1, g2, edge_label)

    best = set()
    stop_at = min(len(g1), len(g2)) * stop_at_fraction
    for (s, n1, n2) in hypotheses:
        if s > min_score:
            return best
        to_check = _get_next(g1, g2, n1, n2, edge_label)
        current = {(n1, n2)}
        current_1 = {n1}
        current_2 = {n2}
        while len(to_check):
            n1, n2 = to_check.pop(0)
            if n1 in current_1 or n2 in current_2:
                continue
            nexts = _get_next(g1, g2, n1, n2, edge_label)
            to_check += nexts
            current.add((n1, n2))
            current_1.add(n1)
            current_2.add(n2)
        if len(current) > stop_at:
            return current
        if len(current) > len(best):
            best = current
    return best

def tonioni_mcs(g1, g2, edge_label = 'dir', min_score = -0.2):
    def find_solution(hypo, cmax):
        current = set()
        while len(hypo):
            s, n1, n2 = hypo[0]
            if s > min_score:
                return current
            current.add((n1, n2))
            nxt = _get_next(g1, g2, n1, n2, edge_label)
            next_hypo = []
            found_1 = set()
            found_2 = set()
            non_mutex_hypotheses = 0
            for s, x1, x2 in hypo[1:]:
                if x1 == n1 or x2 == n2: continue # discard hypotheses
                if (x1, x2) in nxt: s -= 1 # increase scores
                if x1 not in found_1 and x2 not in found_2: # calculate bound
                    non_mutex_hypotheses += 1
                found_1.add(x1)
                found_2.add(x2)
                next_hypo.append((s, x1, x2))
            if len(current) + non_mutex_hypotheses < cmax: #TODO: "Penalize disconnected subgraphs exhibiting differing distances"
                return current
            hypo = sorted(next_hypo)
        return current

    hypotheses = build_hypotheses(g1, g2)

    best = set()
    for i in range(len(hypotheses)):
        if len(best) > len(hypotheses[i:]): return best # no chance of improving anymore
        sol = find_solution(hypotheses[i:], len(best))
        if len(sol) > len(best): #TODO: "Penalize disconnected subgraphs exhibiting differing distances"
            best = sol

    return best

def _project(homography, x, y):
    res = torch.matmul(homography, torch.tensor([x, y, 1], dtype=torch.float))
    return res[:2] / res[2]

def finalize_via_ransac(solution, b1, b2, l1, l2, reproj_threshold = 10, iou_threshold = 0.5):
    nodes1, nodes2 = (list(l) for l in zip(*solution))
    boxes1 = b1[nodes1]
    boxes2 = b2[nodes2]
    points1 = torch.cat((boxes1[:, :2], boxes1[:, 2:]))
    points2 = torch.cat((boxes2[:, :2], boxes2[:, 2:]))
    homography, _ = findHomography(points1, points2, method=RANSAC, ransacReprojThreshold=reproj_threshold)

    expected_positions = torch.stack(
        [torch.cat((_project(homography, x1, y1), _project(homography, x2, y2))) for x1, y1, x2, y2 in b1]
    )

    labels = set(l1) & set(l2)
    matched_expected = torch.full(len(expected_positions), False)

    for lbl in labels: # Find expected w/ matching detections
        expected_indices = l1 == lbl
        reverse_expected = torch.where(expected_indices)[0]
        b2_indices = b2 == lbl
        matched_b2 = torch.full(b2_indices.sum(), False)

        lbl_ious = tvops.box_iou(expected_positions[expected_indices], b2[b2_indices])
        sorted_iou, sort_idx = torch.sort(lbl_ious, dim=1, descending=True)
        for i, (ious, idxs) in enumerate(zip(sorted_iou, sort_idx)):
            for iou, idx in zip(ious, idxs):
                if iou < iou_threshold: break
                if matched_b2[idx]: continue # TODO: We could match with the best instead of with the first
                matched_b2[idx] = True
                matched_expected[reverse_expected[i]] = True

    missing_expected = torch.where(matched_expected == False)[0]
    missing_positions = expected_positions[missing_expected]
    missing_labels = l1[missing_expected]
    return matched_expected, missing_expected, missing_positions, missing_labels
