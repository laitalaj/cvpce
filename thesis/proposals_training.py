import time
import os
from os import path

import torch
from torch import distributed as dist
from torch import nn
from torch import optim as topt
from torch.utils.data import DataLoader, distributed as distutils

from . import datautils
from . import utils
from .models import proposals

def print_time():
    print(f'-- {time.asctime(time.localtime())} --')

def save_pictures(out_path, name, model, img):
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

# TODO: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
def train_proposal_generator(gpu, dataset, output_path, batch_size=1, num_workers=2, epochs=1, gpus=1):
    def checkpoint():
        print(f'Saving results for test image at iteration {i}...')
        img_name = f'{i:05d}'
        save_pictures(output_path, img_name, model, test_image)

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
        print('Saving losses and batch times...')
        torch.save({
            'class_loss': torch.tensor(class_losses),
            'reg_loss': torch.tensor(reg_losses),
            'gauss_loss': torch.tensor(gauss_losses),
            'batch_times': torch.tensor(batch_times),
        }, path.join(output_path, f'stats_{e}.pickle'))
        print(f'Saving model after epoch {e}...')
        out = path.join(output_path, f'epoch_{e}.tar')
        save_model(out, model, optimizer, scheduler, epoch=e)

        print(f'Epoch {e} finished!')
        print_time()

    torch.cuda.set_device(gpu)
    model = proposals.gln().cuda()

    if gpus > 1:
        dist.init_process_group(
            backend='nccl', init_method=f'file://{utils.dist_init_file()}',
            world_size=gpus, rank=gpu
        )
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer = topt.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001) # RetinaNet parameters
    scheduler = topt.lr_scheduler.MultiStepLR(optimizer, milestones=[60_000//len(dataset), 80_000//len(dataset)], gamma=0.1) # RetinaNet learning rate adjustments

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
        print(f'Training for {epochs} epochs, starting now!')
    while e < epochs:
        for batch in loader:
            images, targets = batch.cuda(non_blocking = True)

            if first: batch_start = time.time()

            optimizer.zero_grad()

            loss = model(images, targets)

            total_loss = loss['classification'] + loss['bbox_regression'] + loss['gaussian'] # todo: scaling (though Kant has 1 for all of these)
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

            if first and i % 200 == 0:
                checkpoint()

            i += 1

        scheduler.step()
        if first: epoch_checkpoint()
        e += 1

