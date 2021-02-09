import os
from functools import reduce
from os import path

import torch
from torch import optim as topt
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

from . import datautils, utils
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

class ClassificationTrainingOptions:
    dataset = None
    disciminatorset = None
    output_path = None

    load_gan = None

    min_margin = 0.05 # Numbers from Tonioni's paper
    max_margin = 0.5

    batch_size = 4
    num_workers = 8

    epochs = 1
    checkpoint_interval = 200
    sample_indices = [4096, 4097, 4098, 128, 256, 5000, 6000, 7000, 8000]

    gpus = 1

    def validate(self):
        assert self.dataset is not None, "Dataset must be set"
        assert self.disciminatorset is not None, "Discriminatorset must be set"
        assert self.output_path is not None, "Output path must be set"

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
            total += result
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
        result = utils.scale_from_tanh(model(utils.scale_to_tanh(img[None].cuda()))[0])
        utils.save_multiple([img, result.cpu(), target], path.join(out_path, f'{name}.png'))
    model.train()

def save_dihe_picture(out_path, name, embedder, generator, imgs):
    embedder.eval()
    generator.eval()
    with torch.no_grad():
        fakes = generator(utils.scale_to_tanh(imgs))
        emb_fakes = embedder(utils.scale_from_tanh(fakes))
        emb_reals = embedder(imgs)
        utils.save_emb(path.join(out_path, f'{name}.png'), imgs, emb_reals, fakes, emb_fakes)
    embedder.train()
    generator.train()

def save_gan_state(out, generator, gen_optimizer, discriminator, disc_optimizer, iteration, epoch, distributed=False):
    if distributed: model = model.module
    torch.save({
        GENERATOR_STATE_DICT_KEY: generator.state_dict(),
        GEN_OPTIMIZER_STATE_DICT_KEY: gen_optimizer.state_dict(),
        DISCRIMINATOR_STATE_DICT_KEY: discriminator.state_dict(),
        DISC_OPTIMIZER_STATE_DICT_KEY: disc_optimizer.state_dict(),
        ITERATION_KEY: iteration,
        EPOCH_KEY: epoch,
    }, out)

def save_embedder_state(out, model, optimizer, iteration, epoch, distributed=False):
    if distributed: model = model.module
    torch.save({
        EMBEDDER_STATE_DICT_KEY: model.state_dict(),
        EMB_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
        ITERATION_KEY: iteration,
        EPOCH_KEY: epoch,
    }, out)

def pretrain_gan(options): # TODO: Multi-GPU
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

    generator = classification.unet_generator().cuda() # Learning rates from the DIHE paper
    discriminator = classification.patchgan_discriminator().cuda()

    gen_opt = topt.Adam(generator.parameters(), 1e-5)
    disc_opt = topt.Adam(discriminator.parameters(), 1e-5)

    gen_loader = DataLoader(options.dataset,
        batch_size=options.batch_size, num_workers=options.num_workers,
        collate_fn=datautils.gp_collate_fn, pin_memory=True,
        shuffle=True
    )
    disc_loader = DiscriminatorLoader(options)

    test_image, _, _ = options.dataset[options.sample_indices[0]]
    target_image = options.discriminatorset[options.sample_indices[0] % len(options.discriminatorset)]

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
            loss_regularization = -zncc(fake, gen_batch) # negation: correlation of 1 is the best possible value, correlation of -1 the worst
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

def train_dihe(options): # TODO: Multi-GPU
    def checkpoint():
        print(f'Saving results for test images at iteration {i}...')
        img_name = f'{i:05d}'
        save_dihe_picture(options.output_path, img_name, embedder, generator, test_images.cuda())

        print(f'Saving model and optimizer states...')
        previous_name = 'previous_gan_checkpoint'
        current_name = 'gan_checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_gan_state(current_path, generator, gen_opt, discriminator, disc_opt, i, e)

        previous_name = 'previous_embedder_checkpoint'
        current_name = 'embedder_checkpoint'
        previous_path = path.join(options.output_path, f'{previous_name}.tar')
        current_path = path.join(options.output_path, f'{current_name}.tar')
        if path.exists(current_path):
            os.replace(current_path, previous_path)
        save_embedder_state(current_path, embedder, emb_opt, i, e)

        print('Checkpoint!')
        utils.print_time()

    assert options.load_gan is not None, 'DIHE training should have a pretrained GAN'

    embedder = classification.macvgg_embedder().cuda()
    generator = classification.unet_generator().cuda()
    discriminator = classification.patchgan_discriminator().cuda()

    gan_state = torch.load(options.load_gan)
    generator.load_state_dict(gan_state[GENERATOR_STATE_DICT_KEY])
    discriminator.load_state_dict(gan_state[DISCRIMINATOR_STATE_DICT_KEY])

    emb_opt = topt.Adam(embedder.parameters(), 1e-6) # Learning rates from the DIHE paper
    gen_opt = topt.Adam(generator.parameters(), 1e-5)
    disc_opt = topt.Adam(discriminator.parameters(), 1e-5)
    gen_opt.load_state_dict(gan_state[GEN_OPTIMIZER_STATE_DICT_KEY])
    disc_opt.load_state_dict(gan_state[DISC_OPTIMIZER_STATE_DICT_KEY])

    del gan_state

    train_loader = DataLoader(options.dataset,
        batch_size=options.batch_size * 2, num_workers=options.num_workers, # batch size * 2: get both anchors and negatives
        collate_fn=datautils.gp_collate_fn, pin_memory=True,
        shuffle=True
    )
    disc_loader = DiscriminatorLoader(options)

    test_images = torch.stack([options.dataset[img_index][0] for img_index in options.sample_indices])

    i = 0
    for e in range(options.epochs):
        for batch, gen_batch, hierarchies in train_loader:
            block_size = len(batch) // 2
            if block_size == 0:
                print(f'Got zero block size at iteration {i}, skipping!')
                continue

            disc_batch = disc_loader.get_batch(block_size)

            batch = batch.cuda(non_blocking = True)
            gen_batch = gen_batch[:block_size].cuda(non_blocking = True)
            disc_batch = utils.scale_to_tanh(disc_batch.cuda(non_blocking = True))

            positives = batch[:block_size]
            negatives = batch[block_size:block_size*2]
            pos_hier = hierarchies[:block_size]
            neg_hier = hierarchies[block_size:block_size*2]

            fake = generator(gen_batch)

            disc_opt.zero_grad()
            pred_fake = discriminator(fake.detach())
            pred_real = discriminator(disc_batch)
            loss_fake = nnf.binary_cross_entropy(pred_fake, torch.zeros_like(pred_fake))
            loss_real = nnf.binary_cross_entropy(pred_real, torch.ones_like(pred_real))
            loss_total = loss_fake + loss_real
            loss_total.backward()
            disc_opt.step()


            emb_opt.zero_grad()
            anchor_emb = embedder(utils.scale_from_tanh(fake.detach()))
            positive_emb = embedder(positives)
            negative_emb = embedder(negatives)
            loss = hierarchial_loss(anchor_emb, positive_emb, negative_emb, pos_hier, neg_hier, options.min_margin, options.max_margin)
            loss.backward()
            emb_opt.step()


            gen_opt.zero_grad()
            pred_fake = discriminator(fake)
            positive_emb = embedder(positives)
            fake_emb = embedder(utils.scale_from_tanh(fake))
            loss_adv = nnf.binary_cross_entropy(pred_fake, torch.ones_like(pred_fake))
            loss_regularization = -zncc(fake, positives) # negation: correlation of 1 is the best possible value, correlation of -1 the worst
            loss_emb = -distance(fake_emb, positive_emb).mean()
            loss_total = loss_adv + loss_regularization + 0.1 * loss_emb # weighting from Tonioni
            loss_total.backward()
            gen_opt.step()

            if i % 50 == 0:
                print(f'batch:{i}\tE:{loss:.4f}\tD[real:{loss_real:.4f}\tfake:{loss_fake:.4f}]\tG[adv:{loss_adv:.4f}\treg:{loss_regularization:.4f}\temb:{loss_emb:.4f}]')

            # keepin it clean
            del disc_batch, batch, positives, negatives, fake, pred_fake, pred_real
            del loss, loss_fake, loss_real, loss_adv, loss_regularization, loss_emb, loss_total
            del anchor_emb, positive_emb, negative_emb, fake_emb

            if i % options.checkpoint_interval == 0:
                checkpoint()

            i += 1

    checkpoint()
