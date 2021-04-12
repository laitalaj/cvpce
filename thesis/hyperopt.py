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

    options.apply_hyperopt_config(config)

    proposals_training.train_proposal_generator(0, options)
