import time
import os
from os import path

import torch
from torch import distributed as dist
from torch import nn
from torch import optim as topt
from torch.utils.data import DataLoader, distributed as distutils

from . import datautils, proposals_eval, utils
from .models import proposals

def print_time():
    print(f'-- {time.asctime(time.localtime())} --')

def save_pictures(out_path, name, model, img, distributed=False):
    if distributed: model = model.module # unwrap the actual model underlying DDP as suggested in https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    model.eval()
    with torch.no_grad():
        results = model(img[None].cuda())[0]
        detections_all = utils.recall_tensor(results['boxes'])
        utils.save(img, path.join(out_path, f'{name}_all.png'), detections=detections_all)
        detections_gt_05 = detections_all[utils.recall_tensor(results['scores'] > .5)]
        utils.save(img, path.join(out_path, f'{name}_gt_05.png'), detections=detections_gt_05)
        utils.save(results['gaussians'].cpu(), path.join(out_path, f'{name}_gaussians.png'))
    model.train()

def save_model(out, model, optimizer, scheduler, **extra):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        **extra
    }, out)

def evaluate(model, dataset, batch_size, num_workers, threshold=.75, distributed=False):
    if distributed: model = model.module # unwrap the actual model underlying DDP as suggested in https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    model.eval()
    res = proposals_eval.evaluate_gln_async(model, dataset, thresholds=(threshold,), batch_size=batch_size, num_workers=num_workers, num_metric_processes=num_workers, plots=False)
    model.train()
    return res[threshold]

# TODO: check if checkpoint works correctly: should the underlying modules state dict be saved instead of the wrappers? why are the gaussians blank?
# see: https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
def train_proposal_generator(gpu, dataset, evalset, output_path, batch_size=1, num_workers=2, epochs=1, gpus=1, checkpoint_interval = 200):
    def checkpoint():
        print(f'Saving results for test image at iteration {i}...')
        img_name = f'{i:05d}'
        save_pictures(output_path, img_name, model, test_image, distributed=gpus > 1)

        print(f'Saving model and optimizer state...')
        previous_name = 'previous_checkpoint'
        current_name = 'checkpoint'
        previous_path = path.join(output_path, f'{previous_name}.tar')
        current_path = path.join(output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_model(current_path, model, optimizer, scheduler, epoch=e, iteration=i)

        print('Checkpoint!')
        print_time()

    def epoch_checkpoint():
        old_epoch = e - 2
        old_path = path.join(output_path, f'stats_{old_epoch}.pickle')
        if path.exists(old_path):
            print(f'Deleting old losses and batch times (from epoch {old_epoch})...')
            os.remove(old_path)

        print('Saving losses and batch times...')
        torch.save({
            'class_loss': torch.tensor(class_losses),
            'reg_loss': torch.tensor(reg_losses),
            'gauss_loss': torch.tensor(gauss_losses),
            'batch_times': torch.tensor(batch_times),
        }, path.join(output_path, f'stats_{e}.pickle'))

        print('Evaluating...')
        stats = evaluate(model, evalset, batch_size, num_workers, distributed=gpus > 1)
        if stats['ap'] <= best['ap']:
            print(f'No improvement in epoch {e} ({best["ap"]} at epoch {best["epoch"]} >= {stats["ap"]}')
            print(f'-> Not saving the model! Epoch {e} finished!')
            print_time()
            return best

        print(f'Improvement! Previous best: {best["ap"]} at epoch: {best["epoch"]}; Now {stats["ap"]} (epoch {e})')
        stats['epoch'] = e
        print(f'Saving model at epoch {e}...')
        out = path.join(output_path, f'epoch_{e}.tar')
        save_model(out, model, optimizer, scheduler, epoch=e, iteration=i, stats=stats)

        print(f'Epoch {e} finished!')
        print_time()
        return stats

    torch.cuda.set_device(gpu)
    model = proposals.gln().cuda()

    if gpus > 1:
        dist.init_process_group(
            backend='nccl', init_method=f'file://{utils.dist_init_file()}',
            world_size=gpus, rank=gpu
        )
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer = topt.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0001) # RetinaNet parameters w/ 1/5 lr
    scheduler = topt.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # Halve learning rate every 10 epochs, loosely inspired by RetinaNet

    test_image, _ = dataset[0]
    sampler = distutils.DistributedSampler(dataset, num_replicas=gpus, rank=gpu) if gpus > 1 else None
    loader = DataLoader(dataset,
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=datautils.sku110k_collate_fn, pin_memory=True,
        shuffle=(gpus == 1), sampler=sampler
    )

    e = 0
    i = 0
    first = gpu == 0
    if first:
        class_losses = []
        reg_losses = []
        gauss_losses = []
        batch_times = []
        best = {'epoch': -1, 'ap': 0.0}
        print(f'Training for {epochs} epochs, starting now!')
    while e < epochs:
        for batch in loader:
            images, targets = batch.cuda(non_blocking = True)

            if first: batch_start = time.time()

            optimizer.zero_grad()

            loss = model(images, targets)

            total_loss = loss['classification'] + loss['bbox_regression'] + loss['gaussian'] # todo: scaling (though Kant has 1 for all of these)
            if first and total_loss > 100:
               print(f'!!! Exploded loss at iteration {i}: {loss}')
            total_loss.backward()
            optimizer.step()

            if first:
                batch_end = time.time()
                elapsed = batch_end - batch_start
                class_losses.append(loss['classification'].item())
                reg_losses.append(loss['bbox_regression'].item())
                gauss_losses.append(loss['gaussian'].item())
                batch_times.append(elapsed)

            if first and i % 50 == 0:
                print(f'batch:{i:05d}\t{elapsed:.4f}s\tclass:{class_losses[-1]:.4f}\treg:{reg_losses[-1]:.4f}\tgauss:{gauss_losses[-1]:.4f}')

            del total_loss, loss, batch, images, targets # manual cleanup to get the most out of GPU memory

            if first and i % checkpoint_interval == 0:
                checkpoint()
            if i % checkpoint_interval == 0 and gpus > 1:
                dist.barrier() # wait for rank 0 to checkpoint

            i += 1

        scheduler.step()
        if first: best = epoch_checkpoint()
        if gpus > 1: dist.barrier() # wait for rank 0 to eval and save
        e += 1

