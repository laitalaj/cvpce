from . import datautils, proposals_training

def gln(config, imgs, annotations, eval_annotations, skip, batch_size, dataloader_workers, epochs):
    dataset = datautils.SKU110KDataset(imgs, annotations, skip=skip, gauss_generate_method=datautils.generate_via_kant_method, gauss_join_method=datautils.join_via_replacement)
    evalset = datautils.SKU110KDataset(imgs, eval_annotations, skip=skip, include_gaussians=False)

    options = proposals_training.ProposalTrainingOptions()
    options.dataset = dataset
    options.evalset = evalset
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs

    options.optimizer_lr = config['lr']
    options.optimizer_decay = config['decay']
    options.optimizer_momentum = config['momentum']
    options.lr_multiplier = config['multiplier']

    options.scale_class = config['scale_class']
    options.scale_bbox = config['scale_bbox']
    options.scale_gaussian = config['scale_gaussian']

    options.hyperopt = True

    proposals_training.train_proposal_generator(0, options)
