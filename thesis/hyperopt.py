from . import datautils, proposals_training

def gln(config, imgs, annotations, eval_annotations, skip, batch_size, dataloader_workers, epochs):
    if config['tanh']:
        dataset = datautils.SKU110KDataset(imgs, annotations, skip=skip, tanh=True,
            gauss_generate_method=datautils.generate_via_simple_and_scaled, gauss_join_method=datautils.join_via_max)
    else:
        dataset = datautils.SKU110KDataset(imgs, annotations, skip=skip,
            gauss_generate_method=datautils.generate_via_kant_method, gauss_join_method=datautils.join_via_replacement)

    evalset = datautils.SKU110KDataset(imgs, eval_annotations, skip=skip, include_gaussians=False)

    options = proposals_training.ProposalTrainingOptions()
    options.dataset = dataset
    options.evalset = evalset
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs

    options.tanh = config['tanh']

    options.optimizer_lr = config['lr']
    options.optimizer_decay = config['decay']
    options.optimizer_momentum = config['momentum']
    options.lr_multiplier = config['multiplier']

    options.scale_class = config['scale_class']
    options.scale_gaussian = config['scale_gaussian']

    thresh_min = -1 if config['tanh'] else 0
    thresh_scale = 2 if config['tanh'] else 1
    thresh_low = thresh_min + config['gauss_loss_neg_thresh'] * thresh_scale
    thresh_high = thresh_low + (1 - config['gauss_loss_neg_thresh']) * thresh_scale * config['gauss_loss_pos_thresh']
    options.gaussian_loss_params = {'tanh': config['tanh'], 'negative_threshold': thresh_low, 'positive_threshold': thresh_high}
    print(f'Gaussian loss params: {options.gaussian_loss_params}')

    options.hyperopt = True

    proposals_training.train_proposal_generator(0, options)
