import os
from functools import reduce
from os import path

import torch
from torch import distributed as dist
from torch import optim as topt
from torch import nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader, distributed as distutils
from ray import tune

from . import datautils, utils, classification_eval
from .models import classification
from .models.classification import distance

EMBEDDER_STATE_DICT_KEY = 'model_state_dict'
GENERATOR_STATE_DICT_KEY = 'gen_state_dict'
DISCRIMINATOR_STATE_DICT_KEY = 'disc_state_dict'
EMB_OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
GEN_OPTIMIZER_STATE_DICT_KEY = 'gen_optimizer_state_dict'
DISC_OPTIMIZER_STATE_DICT_KEY = 'disc_optimizer_state_dict'
EPOCH_KEY = 'epoch'
ITERATION_KEY = 'iteration'
BEST_STATS_KEY = 'stats'

class ClassificationTrainingOptions:
    def __init__(self):
        self.dataset = None
        self.discriminatorset = None
        self.evaldata = None
        self.evalset = None
        self.output_path = None

        self.load_encoder = None
        self.load_gan = None

        self.min_margin = 0.05 # Numbers from Tonioni's paper
        self.max_margin = 0.5
        self.enc_lr = 1e-6

        self.enc_multiplier = 1.0

        self.batchnorm = True
        self.masks = False

        self.batch_size = 4
        self.num_workers = 8

        self.epochs = 1
        self.checkpoint_interval = 200
        self.eval_interval = 1
        self.sample_indices = [4096, 4097, 4098, 128, 256, 5000, 6000, 7000, 8000]

        self.gpus = 1

        self.hyperopt = False
    def apply_hyperopt_config(self, config):
        self.batchnorm = config.get('batchnorm', self.batchnorm)
        self.enc_multiplier = config.get('enc_multiplier', self.enc_multiplier)
        self.enc_lr = config.get('enc_lr', self.enc_lr)
        self.hyperopt = True
    def validate(self, pretraining = False):
        assert self.dataset is not None, "Dataset must be set"
        assert self.discriminatorset is not None, "Discriminatorset must be set"
        if not self.hyperopt:
            assert self.output_path is not None, "Output path must be set"
        if not pretraining:
            assert self.load_gan is not None, 'DIHE training should have a pretrained GAN'
            assert self.evalset is not None, 'DIHE training should have a evaluation set'
        if self.evaldata is None:
            self.evaldata = self.dataset


class DiscriminatorLoader:
    def __init__(self, options):
        self.max_size = options.batch_size
        self.loader = DataLoader(options.discriminatorset,
            batch_size=options.batch_size, num_workers=options.num_workers,
            pin_memory=True, shuffle=True
        )
        self.iter = self.generator()
    def generator(self):
        while True:
            for batch in self.loader:
                yield batch
    def get_batch(self, size):
        assert size <= self.max_size, 'DiscriminatorLoader can give batches w/ max size = options.batch_size'

        batch = next(self.iter)
        if size > len(batch): # Didn't get a full batch due to us being at the end of discriminator dataset
            batch = next(self.iter)
        return batch[:size]

class LossMonitor:
    def __init__(self):
        # encoder
        self.loss_dihe = []

        # generator
        self.loss_adv = []
        self.loss_reg = []
        self.loss_emb = []

        # discriminator
        self.loss_real = []
        self.loss_fake = []

    def record_encoder(self, loss_dihe):
        self.loss_dihe.append(loss_dihe.item())

    def record_generator(self, loss_adv, loss_reg, loss_emb):
        self.loss_adv.append(loss_adv.item())
        self.loss_reg.append(loss_reg.item())
        self.loss_emb.append(loss_emb.item())

    def record_discriminator(self, loss_real, loss_fake):
        self.loss_real.append(loss_real.item())
        self.loss_fake.append(loss_fake.item())

    def save(self, path):
        torch.save({
            'dihe_loss': torch.tensor(self.loss_dihe),

            'adv_loss': torch.tensor(self.loss_adv),
            'reg_loss': torch.tensor(self.loss_reg),
            'emb_loss': torch.tensor(self.loss_emb),

            'real_loss': torch.tensor(self.loss_real),
            'fake_loss': torch.tensor(self.loss_fake),
        }, path)

def loaders_and_test_images(gpu, options):
    train_sampler = distutils.DistributedSampler(options.dataset, num_replicas=options.gpus, rank=gpu) if options.gpus > 1 else None
    train_loader = DataLoader(options.dataset,
        batch_size=options.batch_size * 2, num_workers=options.num_workers, # batch size * 2: get both anchors and negatives
        collate_fn=datautils.gp_annotated_collate_fn, pin_memory=True,
        shuffle=(options.gpus == 1), sampler=train_sampler
    )

    # not using distributed sampling here as it's just an auxilliary set for the discriminator and the selection of boxes is random anyway
    # might take it into use later though
    disc_loader = DiscriminatorLoader(options)

    test_images, gen_test_images = zip(*[options.evaldata[img_index % len(options.evaldata)][:2] for img_index in options.sample_indices])
    test_images = torch.stack(test_images)
    gen_test_images = torch.stack(gen_test_images)

    return train_loader, disc_loader, train_sampler, test_images, gen_test_images

def zncc(images, templates):
    assert len(images.shape) == 4, 'Expecting images with shape (idx, channels, height, width)'
    assert images.shape == templates.shape, 'images.shape must match templates.shape'

    istd, imean = torch.std_mean(images, (2, 3))
    tstd, tmean = torch.std_mean(templates, (2, 3))
    total = 0
    for i_idx, (img, tmpl) in enumerate(zip(images, templates)):
        for c_idx, (i_chan, t_chan) in enumerate(zip(img, tmpl)):
            result = (i_chan - imean[i_idx][c_idx]) * (t_chan - tmean[i_idx][c_idx])
            result = result.sum() / (istd[i_idx][c_idx] * tstd[i_idx][c_idx])
            total = total + result
    return total / (reduce(lambda acc, val: acc * val, images.shape))

def masked_zncc(images, templates_with_masks):
    assert len(images.shape) == 4, 'Expecting images with shape (idx, channels, height, width)'

    total = 0
    for img, tmpl in zip(images, templates_with_masks):
        mask = tmpl[3] == 0
        for i_chan, t_chan in zip(img, tmpl):
            masked_ichan = i_chan[mask]
            masked_tchan = t_chan[mask]
            istd, imean = torch.std_mean(masked_ichan)
            tstd, tmean = torch.std_mean(masked_tchan)
            result = (masked_ichan - imean) * (masked_tchan - tmean)
            result = result.sum() / (istd * tstd)
            total = total + result
    return total / (reduce(lambda acc, val: acc * val, images.shape))

def hierarchy_similarity(positives, negatives):
    assert len(positives) == len(negatives), 'Anchors and negatives should be of the same length'

    similarity = torch.empty(len(positives), dtype=torch.float)
    for i, (positive, negative) in enumerate(zip(positives, negatives)):
        scored = False
        for j, p in enumerate(positive):
            if j >= len(negative) or p != negative[j]:
                similarity[i] = j / len(positive)
                scored = True
                break
        if not scored:
            similarity[i] = 1
    return similarity

def hierarchial_loss(anchor_emb, positive_emb, negative_emb, positive_hier, negative_hier, min_margin, max_margin):
    positive_dist = distance(anchor_emb, positive_emb)
    negative_dist = distance(anchor_emb, negative_emb)
    similarity = hierarchy_similarity(positive_hier, negative_hier).to(device = anchor_emb.device)
    margin = min_margin + (1 - similarity) * (max_margin - min_margin)
    loss = torch.clamp(positive_dist - negative_dist + margin, min=0)
    return loss.mean()

def save_gan_picture(out_path, name, model, img, target, distributed=False):
    if distributed: model = model.module # unwrap the actual model underlying DDP as suggested in https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
    model.eval()
    with torch.no_grad():
        result = utils.scale_from_tanh(model(img[None].cuda())[0])
        utils.save_multiple([utils.scale_from_tanh(img[:3]), result.cpu(), target], path.join(out_path, f'{name}.png'))
    model.train()

def save_dihe_picture(out_path, name, embedder, generator, imgs, gen_imgs, distributed=False):
    if distributed:
        embedder = embedder.module
        generator = generator.module

    embedder.eval()
    generator.eval()
    with torch.no_grad():
        fakes = generator(gen_imgs)
        emb_fakes = embedder(fakes)
        emb_reals = embedder(imgs)
        utils.save_emb(path.join(out_path, f'{name}.png'), utils.scale_from_tanh(imgs), emb_reals, utils.scale_from_tanh(fakes), emb_fakes)
    embedder.train()
    generator.train()

def save_gan_state(out, generator, gen_optimizer, discriminator, disc_optimizer, iteration, epoch, distributed=False):
    if distributed:
        generator = generator.module
        discriminator = discriminator.module
    torch.save({
        GENERATOR_STATE_DICT_KEY: generator.state_dict(),
        GEN_OPTIMIZER_STATE_DICT_KEY: gen_optimizer.state_dict(),
        DISCRIMINATOR_STATE_DICT_KEY: discriminator.state_dict(),
        DISC_OPTIMIZER_STATE_DICT_KEY: disc_optimizer.state_dict(),
        ITERATION_KEY: iteration,
        EPOCH_KEY: epoch,
    }, out)

def save_embedder_state(out, model, optimizer, iteration, epoch, best, distributed=False):
    if distributed: model = model.module
    torch.save({
        EMBEDDER_STATE_DICT_KEY: model.state_dict(),
        EMB_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
        ITERATION_KEY: iteration,
        EPOCH_KEY: epoch,
        BEST_STATS_KEY: best,
    }, out)

def evaluate_dihe(model, options, distributed, verbose = True):
    if distributed: model = model.module
    model.eval()
    accuracy = classification_eval.eval_dihe(model, options.evaldata, options.evalset, options.batch_size, options.num_workers, verbose=verbose)[1]
    model.train()
    return accuracy

def pretrain_gan(options):
    def checkpoint():
        print(f'Saving results for test image at iteration {i}...')
        img_name = f'{i:05d}'
        save_gan_picture(options.output_path, img_name, generator, test_image, target_image)

        print(f'Saving model and optimizer states...')
        previous_name = 'previous_checkpoint'
        current_name = 'checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_gan_state(current_path, generator, gen_opt, discriminator, disc_opt, i, e)

        print('Checkpoint!')
        utils.print_time()

    options.validate(pretraining = True)

    generator = classification.unet_generator(options.masks).cuda() # Learning rates from the DIHE paper
    discriminator = classification.patchgan_discriminator().cuda()

    gen_opt = topt.Adam(generator.parameters(), 1e-5)
    disc_opt = topt.Adam(discriminator.parameters(), 1e-5)

    gen_loader = DataLoader(options.dataset,
        batch_size=options.batch_size, num_workers=options.num_workers,
        collate_fn=datautils.gp_collate_fn, pin_memory=True,
        shuffle=True
    )
    disc_loader = DiscriminatorLoader(options)

    _, test_image, _ = options.dataset[options.sample_indices[0] % len(options.dataset)]
    target_image = options.discriminatorset[options.sample_indices[0] % len(options.discriminatorset)]

    regularization = masked_zncc if options.masks else zncc

    i = 0
    for e in range(options.epochs):
        for _, gen_batch, _ in gen_loader:
            disc_batch = disc_loader.get_batch(len(gen_batch))

            gen_batch = gen_batch.cuda(non_blocking = True)
            disc_batch = utils.scale_to_tanh(disc_batch.cuda(non_blocking = True))

            fake = generator(gen_batch)

            disc_opt.zero_grad()
            pred_fake = discriminator(fake.detach())
            pred_real = discriminator(disc_batch)
            loss_fake = nnf.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_real = nnf.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            loss_total = loss_fake + loss_real
            loss_total.backward()
            disc_opt.step()

            gen_opt.zero_grad()
            pred_fake = discriminator(fake)
            loss_adv = nnf.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
            loss_regularization = -regularization(fake, gen_batch) # negation: correlation of 1 is the best possible value, correlation of -1 the worst
            loss_total = loss_adv + loss_regularization
            loss_total.backward()
            gen_opt.step()

            if i % 50 == 0:
                print(f'batch:{i}\tD[real:{loss_real:.4f}\tfake:{loss_fake:.4f}]\tG[adv:{loss_adv:.4f}\treg:{loss_regularization:.4f}]')
            
            del gen_batch, disc_batch, fake, pred_fake, pred_real, loss_fake, loss_real, loss_adv, loss_regularization, loss_total # Keeping CUDA memory nice and clean

            if i % options.checkpoint_interval == 0:
                checkpoint()

            i += 1

    checkpoint()

def train_dihe(gpu, options): # TODO: Evaluation
    def checkpoint():
        distributed = options.gpus > 1

        print(f'Saving results for test images at iteration {i}...')
        img_name = f'{i:05d}'
        save_dihe_picture(options.output_path, img_name, embedder, generator, test_images.cuda(), gen_test_images.cuda(), distributed=distributed)

        print(f'Saving model and optimizer states...')
        previous_name = 'previous_gan_checkpoint'
        current_name = 'gan_checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_gan_state(current_path, generator, gen_opt, discriminator, disc_opt, i, e, distributed=distributed)

        previous_name = 'previous_embedder_checkpoint'
        current_name = 'embedder_checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_embedder_state(current_path, embedder, emb_opt, i, e, best, distributed=distributed)

        print('Checkpoint!')
        utils.print_time()

    def epoch_checkpoint(final = False):
        old_epoch = e - 2
        old_path = path.join(options.output_path, f'stats_{old_epoch}.pickle')
        if path.exists(old_path):
            print(f'Deleting old losses and batch times (from epoch {old_epoch})...')
            os.remove(old_path)

        print('Saving losses...')
        current_name = f'stats_{e}.pickle'
        current_path = path.join(options.output_path, current_name)
        losses.save(current_path)

        distributed = options.gpus > 1
        if e % options.eval_interval == 0 or final:
            out = path.join(options.output_path, f'epoch_{e}.tar')
            gan_out = path.join(options.output_path, f'epoch_{e}_gan.tar')
            print('Evaluating...')
            accuracy = evaluate_dihe(embedder, options, distributed)
            if accuracy <= best['accuracy']:
                print(f'No improvement in epoch {e} ({best["accuracy"]:.4f} at epoch {best["epoch"]} > {accuracy:.4f}')
                if final:
                    print('-> Saving despite this due to being on the final iteration')
                    save_embedder_state(out, embedder, emb_opt, i, e, best, distributed=distributed)
                    save_gan_state(gan_out, generator, gen_opt, discriminator, disc_opt, i, e, distributed=distributed)
                else:
                    print('-> Not saving the model!')
            else:
                print(f'Improvement! Previous best: {best["accuracy"]:.4f} at epoch: {best["epoch"]}; Now {accuracy:.4f} (epoch {e})')
                stats = {'accuracy': accuracy, 'epoch': e}
                print(f'Saving model at epoch {e}...')
                save_embedder_state(out, embedder, emb_opt, i, e, stats, distributed=distributed)
                save_gan_state(gan_out, generator, gen_opt, discriminator, disc_opt, i, e, distributed=distributed)
                print(f'Epoch {e} finished!')
                return stats

        print(f'Epoch {e} finished!')
        utils.print_time()
        return best

    options.validate()

    torch.cuda.set_device(gpu)

    embedder = classification.macvgg_embedder(model='vgg16_bn' if options.batchnorm else 'vgg16').cuda()
    generator = classification.unet_generator(options.masks).cuda()
    discriminator = classification.patchgan_discriminator().cuda()

    map_location = {'cuda:0': f'cuda:{gpu}'}

    gan_state = torch.load(options.load_gan, map_location=map_location)
    generator.load_state_dict(gan_state[GENERATOR_STATE_DICT_KEY])
    discriminator.load_state_dict(gan_state[DISCRIMINATOR_STATE_DICT_KEY])

    load = options.load_encoder is not None
    if load:
        state = torch.load(options.load_encoder, map_location=map_location)
        embedder.load_state_dict(state[EMBEDDER_STATE_DICT_KEY])

    if options.gpus > 1: # TODO: Might be worth it to check which parts here are actually necessary; did a bunch of changes trying to get DDP working here (broadcast_buffers=False did the trick)
        dist.init_process_group(
            backend='nccl', init_method=f'file://{utils.dist_init_file()}',
            world_size=options.gpus, rank=gpu
        )
        embedder = nn.parallel.DistributedDataParallel(embedder, device_ids=[gpu], process_group=dist.new_group(), broadcast_buffers=False)
        generator = nn.parallel.DistributedDataParallel(generator, device_ids=[gpu], process_group=dist.new_group(), broadcast_buffers=False)
        discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[gpu], process_group=dist.new_group(), broadcast_buffers=False)

    emb_opt = topt.Adam(embedder.parameters(), options.enc_lr)
    gen_opt = topt.Adam(generator.parameters(), 1e-5) # Learning rates from the DIHE paper
    disc_opt = topt.Adam(discriminator.parameters(), 1e-5)

    scheduler = topt.lr_scheduler.MultiplicativeLR(emb_opt, lambda _: options.enc_multiplier, verbose=not options.hyperopt)

    gen_opt.load_state_dict(gan_state[GEN_OPTIMIZER_STATE_DICT_KEY])
    disc_opt.load_state_dict(gan_state[DISC_OPTIMIZER_STATE_DICT_KEY])
    if load:
        emb_opt.load_state_dict(state[EMB_OPTIMIZER_STATE_DICT_KEY])

    del gan_state

    train_loader, disc_loader, train_sampler, test_images, gen_test_images = loaders_and_test_images(gpu, options)

    regularization = masked_zncc if options.masks else zncc

    first = gpu == 0 and not options.hyperopt
    start_epoch = state[EPOCH_KEY] + 1 if load else 0
    end_epoch = start_epoch + options.epochs
    epoch_range = range(start_epoch, end_epoch)
    i = state[ITERATION_KEY] + 1 if load else 0
    if first:
        losses = LossMonitor()
        best = state[BEST_STATS_KEY] if load else {'accuracy': 0.0, 'epoch': -1}
        print(f'Training for {options.epochs} epochs, starting now!')

    if load: del state # everything's been loaded, make sure that this doesn't consume any extra memory

    for e in epoch_range:
        if options.gpus > 1:
            train_sampler.set_epoch(e)

        for batch, gen_batch, hierarchies, _ in train_loader:
            block_size = len(batch) // 2
            if block_size == 0:
                print(f'Got zero block size at iteration {i}, skipping!') #TODO: This might result in failures in a multi-gpu setting :think:
                continue

            disc_batch = disc_loader.get_batch(block_size)

            batch = batch.cuda(non_blocking = True)
            gen_batch = gen_batch[:block_size].cuda(non_blocking = True)
            disc_batch = utils.scale_to_tanh(disc_batch.cuda(non_blocking = True))

            positives = batch[:block_size]
            negatives = batch[block_size:block_size*2]
            pos_hier = hierarchies[:block_size]
            neg_hier = hierarchies[block_size:block_size*2]

            # encoder
            emb_opt.zero_grad()
            fake = generator(gen_batch)
            anchor_emb = embedder(fake)
            positive_emb = embedder(positives)
            negative_emb = embedder(negatives)
            loss = hierarchial_loss(anchor_emb, positive_emb, negative_emb, pos_hier, neg_hier, options.min_margin, options.max_margin)
            loss.backward()
            emb_opt.step()
            if first:
                losses.record_encoder(loss)

            # discriminator
            disc_opt.zero_grad()
            fake = generator(gen_batch)
            pred_fake = discriminator(fake)
            pred_real = discriminator(disc_batch)
            loss_fake = nnf.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_real = nnf.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            loss_total = loss_fake + loss_real
            loss_total.backward()
            disc_opt.step()
            if first:
                losses.record_discriminator(loss_real, loss_fake)

            # generator
            gen_opt.zero_grad()
            fake = generator(gen_batch)
            pred_fake = discriminator(fake)
            positive_emb = embedder(positives)
            fake_emb = embedder(fake)
            loss_adv = nnf.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
            loss_regularization = -regularization(fake, gen_batch) # negation: correlation of 1 is the best possible value, correlation of -1 the worst
            loss_emb = -distance(fake_emb, positive_emb).mean()
            loss_total = loss_adv + loss_regularization + 0.1 * loss_emb # weighting from Tonioni
            loss_total.backward()
            gen_opt.step()
            if first:
                losses.record_generator(loss_adv, loss_regularization, loss_emb)

            if first and i % 50 == 0:
                print(f'batch:{i}\tE:{loss:.4f}\tD[real:{loss_real:.4f}\tfake:{loss_fake:.4f}]\tG[adv:{loss_adv:.4f}\treg:{loss_regularization:.4f}\temb:{loss_emb:.4f}]')

            # keepin it clean
            del disc_batch, batch, positives, negatives, fake, pred_fake, pred_real
            del loss, loss_fake, loss_real, loss_adv, loss_regularization, loss_emb, loss_total
            del anchor_emb, positive_emb, negative_emb, fake_emb

            if first and i % options.checkpoint_interval == 0:
                checkpoint()
            if i % options.checkpoint_interval == 0 and options.gpus > 1:
                dist.barrier()

            i += 1
        
        scheduler.step()
        if first:
            best = epoch_checkpoint(e == end_epoch - 1)
        elif options.hyperopt:
            accuracy = evaluate_dihe(embedder, options, options.gpus > 1, verbose=False)
            tune.report(accuracy = accuracy)
        if options.gpus > 1: dist.barrier()

