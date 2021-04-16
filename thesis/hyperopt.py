from . import datautils, proposals_training, classification_training

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

    options.apply_hyperopt_config(config)

    proposals_training.train_proposal_generator(0, options)

def dihe(config, source_dir, target_imgs, target_annotations, eval_imgs, eval_annotations, load_gan, masks, source_only, target_skip, eval_only, batch_size, dataloader_workers, epochs):
    options = classification_training.ClassificationTrainingOptions()

    options.dataset = datautils.GroceryProductsDataset(source_dir, include_annotations=True, include_masks=masks, only=source_only if len(source_only) else None)
    options.discriminatorset = datautils.TargetDomainDataset(target_imgs, target_annotations, skip=target_skip)
    options.evalset = datautils.GroceryProductsTestSet(eval_imgs, eval_annotations, only=eval_only)

    options.load_gan = load_gan
    options.masks = masks
    options.batch_size = batch_size
    options.num_workers = dataloader_workers
    options.epochs = epochs

    options.apply_hyperopt_config(config)

    classification_training.train_dihe(0, options)
