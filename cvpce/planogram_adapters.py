import json

import torch
import networkx as nx

def _get_object(planogram, graph, node):
    return planogram['objects'][graph.nodes[node]['ogg']]

def _process_dir(d): # Flip for compliance w/ detections
    res = d.upper()
    if 'N' in res:
        return res.replace('N', 'S')
    if 'S' in res:
        return res.replace('S', 'N')
    return res

def read_tonioni_planogram(planogram_path):
    with open(planogram_path, 'r') as planogram_file:
        planogram = json.load(planogram_file)

    g = nx.DiGraph()
    western_nodes = set()
    southern_nodes = set()
    for i, entry in enumerate(planogram['graph']):
        g.add_node(i, ogg=entry['ogg'])
        g.add_edges_from((i, j, {'dir': _process_dir(k)}) for k, j in entry.items() if j >= 0 and k != 'ogg')
        if entry['w'] == -1:
            western_nodes.add(i)
        if entry['n'] == -1: # Flip for compliance w/ detections, TODO: clean this stuff up a bit
            southern_nodes.add(i)

    rows = {w: [] for w in western_nodes}
    cols = {s: [] for s in southern_nodes}
    for w, r in rows.items():
        prev = -1
        nxt = [w]
        while len(nxt):
            if len(nxt) > 1: raise RuntimeError(f'Multiple nodes east from {prev}: {nxt} (file: {planogram_path})')
            nxt = nxt[0]
            g.nodes[nxt]['row'] = w
            r.append(nxt)
            prev = nxt
            nxt = [e for e in g[prev] if g[prev][e]['dir'] == 'E']
    for s, c in cols.items():
        prev = -1
        nxt = [s]
        while len(nxt):
            if len(nxt) > 1: raise RuntimeError(f'Multiple nodes north from {prev}: {nxt} (file: {planogram_path})')
            nxt = nxt[0]
            g.nodes[nxt]['col'] = s
            c.append(nxt)
            prev = nxt
            nxt = [n for n in g[prev] if g[prev][n]['dir'] == 'N']

    row_y = {w: float('-inf') for w in rows}
    col_x = {s: float('-inf') for s in cols}
    for r in rows.values():
        baseline = 0
        x = 0
        for p in r:
            col = g.nodes[p]['col']
            if col_x[col] > float('-inf'):
                baseline = col_x[col] - x
                break
            x += _get_object(planogram, g, p)['width']
        x = baseline
        for p in r:
            col = g.nodes[p]['col']
            col_x[col] = max(x, col_x[col])
            x += _get_object(planogram, g, p)['width']
    for c in cols.values():
        baseline = 0
        y = 0
        for p in c:
            row = g.nodes[p]['row']
            if row_y[row] > float('-inf'):
                baseline = row_y[row] - y
                break
            y += _get_object(planogram, g, p)['height']
        y = baseline
        for p in c:
            row = g.nodes[p]['row']
            row_y[row] = max(y, row_y[row])
            y += _get_object(planogram, g, p)['height']

    for r in rows.values():
        x = col_x[g.nodes[r[0]]['col']] + _get_object(planogram, g, r[0])['width']
        for p in r[1:]:
            col = g.nodes[p]['col']
            if x > col_x[col]:
                col_x[col] = x
            else:
                x = col_x[col]
            x += _get_object(planogram, g, p)['width']
    for c in cols.values():
        y = row_y[g.nodes[c[0]]['row']] + _get_object(planogram, g, c[0])['height']
        for p in c[1:]:
            row = g.nodes[p]['row']
            if y > row_y[row]:
                row_y[row] = y
            else:
                y = row_y[row]
            y += _get_object(planogram, g, p)['height']

    for n, node in g.nodes.items():
        obj = _get_object(planogram, g, n)
        x1 = col_x[node['col']]
        y1 = row_y[node['row']] - obj['height']
        x2 = x1 + obj['width']
        y2 = row_y[node['row']]
        node['pos'] = (x1, y1, x2, y2)

    node_range = range(len(planogram['graph']))
    boxes = torch.tensor([g.nodes[i]['pos'] for i in node_range], dtype=torch.float)
    for i in g:
        label = _get_object(planogram, g, i)['img_path']
        #label = f'{g.nodes[i]["row"]}/{g.nodes[i]["col"]}'
        del g.nodes[i]['pos'], g.nodes[i]['row'], g.nodes[i]['col'], g.nodes[i]['ogg']
        g.nodes[i]['label'] = label.split('.')[0]
    labels = [g.nodes[i]['label'] for i in node_range]

    return boxes, labels, g
