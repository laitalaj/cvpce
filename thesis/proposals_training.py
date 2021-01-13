import time
import os
from os import path

import torch
from torch import nn
from torch import optim as topt
from torch.utils.data import DataLoader

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

def save_model(out, model, optimizer, **extra):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **extra
    }, out)

def train_proposal_generator(dataset, output_path, batch_size=1, num_workers=2, epochs=1, parallel=False):
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
        save_model(current_path, model, optimizer, epoch=e, iteration=i)

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
        save_model(out, model, optimizer, epoch=e)

        print(f'Epoch {e} finished!')
        print_time()

    model = proposals.gln()
    if parallel:
        model = nn.DataParallel(model)
    model.cuda()
    
    optimizer = topt.SGD(model.parameters(), lr=0.001, momentum=0.9) # todo: finetune

    test_image, _ = dataset[0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=datautils.sku110k_collate_fn)

    class_losses = []
    reg_losses = []
    gauss_losses = []
    batch_times = []
    e = 0
    i = 0
    print(f'Training for {epochs} epochs, starting now!')
    while e < epochs:
        for images, targets in loader:
            images = [i.cuda() for i in images]
            targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda(), 'gaussians': t['gaussians'].cuda()} for t in targets]

            batch_start = time.time()

            optimizer.zero_grad()

            loss = model(images, targets)

            total_loss = loss['classification'] + loss['bbox_regression'] + loss['gaussian'] # todo: scaling
            total_loss.backward()
            optimizer.step()

            batch_end = time.time()
            elapsed = batch_end - batch_start

            class_losses.append(loss['classification'].item())
            reg_losses.append(loss['bbox_regression'].item())
            gauss_losses.append(loss['gaussian'].item())
            batch_times.append(elapsed)

            if i % 100 == 0:
                print(f'batch:{i:05d}\t{elapsed:.4f}s\tclass:{class_losses[-1]:.4f}\treg:{reg_losses[-1]:.4f}\tgauss:{gauss_losses[-1]:.4f}')

            del total_loss, loss, images, targets # manual cleanup to get the most out of GPU memory

            if i % 200 == 0:
                checkpoint()

            i += 1

        epoch_checkpoint()
        e += 1

