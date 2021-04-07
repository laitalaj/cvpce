import time
import os
from os import path

from ray import tune
import torch
from torch import distributed as dist
from torch import nn
from torch import optim as topt
from torch.utils.data import DataLoader, distributed as distutils
from torchvision import ops as tvops

from . import datautils, proposals_eval, utils
from .utils import print_time
from .models import proposals

MODEL_STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
SCHEDULER_STATE_DICT_KEY = 'scheduler_state_dict'
EPOCH_KEY = 'epoch'
ITERATION_KEY = 'iteration'
BEST_STATS_KEY = 'stats'

class ProposalTrainingOptions:
    def __init__(self):
        self.dataset = None
        self.evalset = None
        self.output_path = None

        self.load = None
        self.trim_module_prefix = False

        self.optimizer_lr = 0.0025 # RetinaNet parameters w/ 1/4 learning rate
        self.optimizer_momentum = 0.9
        self.optimizer_decay = 0.0001
        self.lr_multiplier = 0.99

        self.scale_class = 1
        self.scale_gaussian = 1

        self.tanh = False
        self.gaussian_loss_params = {}

        self.batch_size = 1
        self.num_workers = 2

        self.epochs = 1
        self.checkpoint_interval = 200

        self.gpus = 1

        self.hyperopt = False
    def validate(self):
        assert self.dataset is not None, "Dataset must be set"
        assert self.evalset is not None, "Evalset must be set"
        assert self.output_path is not None or self.hyperopt, "Output path must be set if not hyperopting"


def optimizer_and_scheduler(model, options):
    optimizer = topt.SGD(model.parameters(), lr=options.optimizer_lr, momentum=options.optimizer_momentum, weight_decay=options.optimizer_decay)
    scheduler = topt.lr_scheduler.MultiplicativeLR(optimizer, lambda _: options.lr_multiplier, verbose=not options.hyperopt) # Slightly decay learning rate after every epoch, loosely inspired by RetinaNet
    return optimizer, scheduler

def loader_and_test_img(gpu, options):
    test_image, _ = options.dataset[0]
    sampler = distutils.DistributedSampler(options.dataset, num_replicas=options.gpus, rank=gpu) if options.gpus > 1 else None
    loader = DataLoader(options.dataset,
        batch_size=options.batch_size, num_workers=options.num_workers,
        collate_fn=datautils.sku110k_collate_fn, pin_memory=True,
        shuffle=(options.gpus == 1), sampler=sampler
    )
    return loader, sampler, test_image

def save_pictures(out_path, name, model, img, distributed=False):
    if distributed: model = model.module # unwrap the actual model underlying DDP as suggested in https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    model.eval()
    with torch.no_grad():
        results = model(img[None].cuda())[0]
        detections_all = utils.recall_tensor(tvops.box_convert(results['boxes'], 'xyxy', 'xywh'))
        utils.save(img, path.join(out_path, f'{name}_all.png'), detections=detections_all)
        detections_gt_05 = detections_all[utils.recall_tensor(results['scores'] > .5)]
        utils.save(img, path.join(out_path, f'{name}_gt_05.png'), detections=detections_gt_05)
        utils.save(results['gaussians'].cpu(), path.join(out_path, f'{name}_gaussians.png'))
    model.train()

def save_state(out, model, optimizer, scheduler, iteration, epoch, best, distributed=False):
    if distributed: model = model.module
    torch.save({
        MODEL_STATE_DICT_KEY: model.state_dict(),
        OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
        SCHEDULER_STATE_DICT_KEY: scheduler.state_dict(),
        ITERATION_KEY: iteration,
        EPOCH_KEY: epoch,
        BEST_STATS_KEY: best,
    }, out)

def evaluate(model, dataset, batch_size, num_workers, threshold=.75, distributed=False):
    if distributed: model = model.module # unwrap the actual model underlying DDP as suggested in https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    model.eval()
    # async evaluation doesn't currently work with multi-GPU training; see https://bugs.python.org/issue39959 ; https://github.com/python/cpython/pull/21516
    res = proposals_eval.evaluate_gln_sync(model, dataset, thresholds=(threshold,), batch_size=batch_size, num_workers=num_workers, plots=False)
    model.train()
    return res[threshold]


def train_proposal_generator(gpu, options):
    def checkpoint():
        print(f'Saving results for test image at iteration {i}...')
        img_name = f'{i:05d}'
        save_pictures(options.output_path, img_name, model, test_image, distributed=options.gpus > 1)

        print(f'Saving model and optimizer state...')
        previous_name = 'previous_checkpoint'
        current_name = 'checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_state(current_path, model, optimizer, scheduler, i, e, best, distributed=options.gpus > 1)

        print('Checkpoint!')
        print_time()

    def epoch_checkpoint(final=False):
        old_epoch = e - 2
        old_path = path.join(options.output_path, f'stats_{old_epoch}.pickle')
        if path.exists(old_path):
            print(f'Deleting old losses and batch times (from epoch {old_epoch})...')
            os.remove(old_path)

        print('Saving losses and batch times...')
        torch.save({
            'class_loss': torch.tensor(class_losses),
            'reg_loss': torch.tensor(reg_losses),
            'gauss_loss': torch.tensor(gauss_losses),
            'batch_times': torch.tensor(batch_times),
        }, path.join(options.output_path, f'stats_{e}.pickle'))

        if e % 3 == 0 or final: # TODO: Make configurable
            out = path.join(options.output_path, f'epoch_{e}.tar')
            print('Evaluating...')
            stats = evaluate(model, options.evalset, options.batch_size, options.num_workers, distributed=options.gpus > 1)
            if stats['ap'] <= best['ap']:
                print(f'No improvement in epoch {e} ({best["ap"]} at epoch {best["epoch"]} >= {stats["ap"]}')
                if final:
                    print('-> Saving despite this due to being on the final iteration')
                    save_state(out, model, optimizer, scheduler, i, e, best, distributed=options.gpus > 1)
                else:
                    print('-> Not saving the model!')
            else:
                print(f'Improvement! Previous best: {best["ap"]} at epoch: {best["epoch"]}; Now {stats["ap"]} (epoch {e})')
                stats['epoch'] = e
                print(f'Saving model at epoch {e}...')
                save_state(out, model, optimizer, scheduler, i, e, stats, distributed=options.gpus > 1)
                return stats

        print(f'Epoch {e} finished!')
        print_time()
        return best

    options.validate()

    load = options.load is not None
    if load:
        state = torch.load(options.load, map_location={'cuda:0': f'cuda:{gpu}'})

    torch.cuda.set_device(gpu)
    model = proposals.gln(tanh = options.tanh, gaussian_loss_params=options.gaussian_loss_params).cuda()
    if load:
        model.load_state_dict(
            utils.trim_module_prefix(state[MODEL_STATE_DICT_KEY]) if options.trim_module_prefix else state[MODEL_STATE_DICT_KEY]
        )

    if options.gpus > 1:
        dist.init_process_group(
            backend='nccl', init_method=f'file://{utils.dist_init_file()}',
            world_size=options.gpus, rank=gpu
        )
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer, scheduler = optimizer_and_scheduler(model, options)
    if load:
        optimizer.load_state_dict(state[OPTIMIZER_STATE_DICT_KEY])
        scheduler.load_state_dict(state[SCHEDULER_STATE_DICT_KEY])


    loader, sampler, test_image = loader_and_test_img(gpu, options)

    first = gpu == 0 and not options.hyperopt
    start_epoch = state[EPOCH_KEY] + 1 if load else 0
    end_epoch = start_epoch + options.epochs
    epoch_range = range(start_epoch, end_epoch)
    i = state[ITERATION_KEY] + 1 if load else 0
    if first:
        class_losses = []
        reg_losses = []
        gauss_losses = []
        batch_times = []
        best = state[BEST_STATS_KEY] if load else {'epoch': -1, 'ap': 0.0}
        print(f'Training for {options.epochs} epochs, starting now!')
    
    if load:
        del state # everything's been loaded, make sure that this doesn't consume any extra memory

    for e in epoch_range:
        if options.gpus > 1:
            sampler.set_epoch(e)

        for batch in loader:
            images, targets = batch.cuda(non_blocking = True)

            if first: batch_start = time.time()

            optimizer.zero_grad()

            loss = model(images, targets)

            total_loss = options.scale_class * loss['classification'] \
                + loss['bbox_regression'] \
                + options.scale_gaussian * loss['gaussian']
            if total_loss > 5000:
                if first:
                    print(f'!!! Exploded loss at iteration {i}: {loss}')
                elif options.hyperopt and gpu == 0:
                    raise RuntimeError(f'Exploded loss at iteration {i}: {loss}')
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

            if first and i % options.checkpoint_interval == 0:
                checkpoint()
            if i % options.checkpoint_interval == 0 and options.gpus > 1 and not options.hyperopt:
                dist.barrier() # wait for rank 0 to checkpoint

            i += 1

        scheduler.step()
        if first: best = epoch_checkpoint(e == end_epoch - 1)
        elif options.hyperopt and gpu == 0:
            stats = evaluate(model, options.evalset, options.batch_size, options.num_workers, distributed=options.gpus > 1)
            tune.report(**stats)
        if options.gpus > 1: dist.barrier() # wait for rank 0 to eval and save
