import re

import torch
from torch.utils import data as tdata

from . import metrics, datautils, utils
from .models import proposals

def _load_gln(save_file, trim_module_prefix):
    state = torch.load(save_file)
    state_dict = state['model_state_dict']
    if trim_module_prefix:
        regex = re.compile(r'^module\.(.*)$')
        state_dict = {regex.match(k).group(1): v for k, v in state_dict.items()}
    model = proposals.gln(pretrained_backbone=False).cuda()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def evaluate_gln(save_file, dataset, thresholds=(.5,), batch_size=1, num_workers=2, plots=True, trim_module_prefix=True):
    model = _load_gln(save_file, trim_module_prefix)

    loader = tdata.DataLoader(dataset,
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=datautils.sku110k_no_gauss_collate_fn, pin_memory=True,
    )
    predictions = []
    targets = []
    confidences = []
    print('Eval start!')
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print(f'{i}...')
        images, batch_targets = batch.cuda(non_blocking = True)
        result = model(images)

        for r, t in zip(result, batch_targets):
            predictions.append(r['boxes'].detach().cpu())
            targets.append(t['boxes'].detach().cpu())
            confidences.append(r['scores'].detach().cpu())

    print('All data passed through model! Calculating metrics...')
    res = metrics.calculate_metrics(targets, predictions, confidences, thresholds)
    print('Metrics calculated!')
    if plots:
        for t in thresholds:
            print(f'Plotting t={t}...')
            metrics.plot_prf(res[t]['raw']['p'], res[t]['raw']['r'], res[t]['raw']['f'])
    print('Eval done!')
    return {thresh: {k: v for k, v in itm.items() if k != 'raw'} for thresh, itm in res.items()}

def evaluate_gln_async(save_file, dataset, thresholds=(.5,), batch_size=1, num_workers=2, num_metric_processes=4, plots=True, trim_module_prefix=True):
    model = _load_gln(save_file, trim_module_prefix)
    loader = tdata.DataLoader(dataset,
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=datautils.sku110k_no_gauss_collate_fn, pin_memory=True,
    )

    queue, pipe = metrics.calculate_metrics_async(processes=num_metric_processes, iou_thresholds=thresholds)
    print('Eval start!')
    for i, batch in enumerate(loader):
        if i % 100 == 0:
            print(f'{i}...')
        images, batch_targets = batch.cuda(non_blocking = True)
        result = model(images)
        for r, t in zip(result, batch_targets):
            queue.put((t['boxes'].detach().cpu(), r['boxes'].detach().cpu(), r['scores'].detach().cpu()))

    print('All data passed through model! Waiting for metric workers...')
    for _ in range(num_metric_processes):
        queue.put(None)
    queue.join()
    print('Starting metric calculation...')
    pipe.send(True)
    res = pipe.recv()
    print('Metrics calculated!')
    if plots:
        for t in thresholds:
            print(f'Plotting t={t}...')
            metrics.plot_prf(res[t]['raw']['p'], res[t]['raw']['r'], res[t]['raw']['f'])
    print('Eval done!')
    return {thresh: {k: v for k, v in itm.items() if k != 'raw'} for thresh, itm in res.items()}
