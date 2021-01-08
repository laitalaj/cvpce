import time
from os import path

import torch
from torch import optim as topt
from torch.utils.data import DataLoader

from . import datautils
from . import utils
from .models import proposals

def train_proposal_generator(img_dir_path, annotation_file_path, output_path, batch_size=1, num_workers=2):
    def print_time():
        print(f'-- {time.asctime(time.localtime())} --')
    def checkpoint():
        print(f'Saving results for test image at iteration {i}...')
        img_name = f'{i:05d}'
        model.eval()
        with torch.no_grad():
            results = model(test_image[None].cuda())[0]
            detections_all = utils.recall_tensor(results['boxes'])
            print(detections_all.shape)
            utils.save(test_image, path.join(output_path, f'{img_name}_predictions_all.png'), detections=detections_all)
            detections_gt_05 = detections_all[utils.recall_tensor(results['scores'] > .5)]
            utils.save(test_image, path.join(output_path, f'{img_name}_predictions_gt_05.png'), detections=detections_gt_05)
            utils.save(results['gaussians'].cpu(), path.join(output_path, f'{img_name}_gaussians.png'))
        model.train()
        print(f'Saving model and optimizer state...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path.join(output_path, f'{img_name}.tar'))
        print('Checkpoint!')
        print_time()

    model = proposals.gln().cuda()
    optimizer = topt.SGD(model.parameters(), lr=0.001, momentum=0.9) # todo: finetune

    sku110k_train = datautils.SKU110KDataset(img_dir_path, annotation_file_path)
    test_image, _ = sku110k_train[0]
    loader = DataLoader(sku110k_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=datautils.sku110k_collate_fn)

    i = 0
    for images, targets in loader:
        images = [i.cuda() for i in images]
        targets = [{'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda(), 'gaussians': t['gaussians'].cuda()} for t in targets]

        optimizer.zero_grad()

        loss = model(images, targets)
        if i % 25 == 0:
            print(i, loss)

        total_loss = loss['classification'] + loss['bbox_regression'] + 0.00001 * loss['gaussian'] # todo: scaling
        total_loss.backward()
        optimizer.step()

        del total_loss, loss, images, targets # manual cleanup to get the most out of GPU memory

        if i % 200 == 0:
            checkpoint()

        i += 1
    
    checkpoint()
