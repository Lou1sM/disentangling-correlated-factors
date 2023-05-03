import re
import os

import fastargs
import numpy as np

import datasets
import dent.models
import dent.losses
import dent.losses.utils
import dent.utils
import parameters
import utils
import utils.visualize

def main(config):
    logger = utils.set_logger(__name__)
    utils.set_seed()
    device = utils.get_device()

    # Set all required folders and files if not explicitly set.
    config, info_dict = utils.set_save_paths_or_restore(config, logger, device)
    #config.summary()

    constraints_filepath = config['constraints.file']
    if constraints_filepath == 'none':
        constraints_filepath = None
    correlations_filepath = config['constraints.correlations_file']
    if correlations_filepath == 'none':
        correlations_filepath = None
    train_loader, log_summary = datasets.get_dataloaders(
        shuffle=True,
        device=device,
        logger=logger,
        return_pairs='pairs' in config['train.supervision'],
        root=config['data.root'],
        k_range=config['data.k_range'],
        constraints_filepath=constraints_filepath,
        correlations_filepath=correlations_filepath)
    config = utils.insert_config('data.img_size',
                                 train_loader.dataset.img_size)
    config = utils.insert_config('data.background_color',
                                 train_loader.dataset.background_color)

    model = dent.model_select(device, img_size=config['data.img_size'])

    if hasattr(model, 'to_optim'):
        optimizer = utils.get_optimizer(model.to_optim)
    else:
        optimizer = utils.get_optimizer(model.parameters())
    scheduler = utils.get_scheduler(optimizer)
    model = model.to(device)
    loss_f = dent.loss_select(device, n_data=len(train_loader.dataset))
    if config['train.supervision'] == 'none':
        trainer = dent.UnsupervisedTrainer(
            model=model, optimizer=optimizer, scheduler=scheduler, loss_f=loss_f,
            device=device, logger=logger, read_dir=info_dict['read_dir'],
            write_dir=info_dict['write_dir'])
    elif 'pairs' in config['train.supervision']:
        infer_k = True
        if config['train.supervision'] == 'pairs_and_shared':
            infer_k = False
        trainer = dent.WeaklySupervisedPairTrainer(
            model=model, optimizer=optimizer, scheduler=scheduler, loss_f=loss_f,
            device=device, logger=logger, read_dir=info_dict['read_dir'],
            write_dir=info_dict['write_dir'], infer_k=infer_k)
    else:
        raise ValueError(f'No trainer [{config["train.supervision"]}] available!')

    trainer.initialize()

    trainer(train_loader)

    z_list = []
    gt_list = []
    for xb,yb in train_loader:
        enc_output = model.encoder(xb.cuda())
        mean,logvar = enc_output['stats_qzx'].unbind(2)
        z_list.append(mean.detach().cpu().numpy())
        gt_list.append(yb.detach().cpu().numpy())
    z_array = np.concatenate(z_list,axis=0)
    gt_array = np.concatenate(gt_list,axis=0)
    #rename_dset_dict = {'shapes3d': '3'}
    #rename_method_dict = {'hfs': 'h'}
    #fpath = f'{rename_dset_dict[bad_dset_name]}{rename_method_dict[bad_method_name]}.npz'
    #np.savez(f"{info_dict['write_dir']}/latents.npz",latents=z_array,gts=gt_array)
    correlations_info = '' if correlations_filepath is None else '_'.join(correlations_filepath.split(':')[1].split('_')[1:])
    fpath =f"3h{correlations_info}.npz"
    while True:
        if not os.path.exists(fpath):
            print(f'saving to {fpath}')
            np.savez(fpath,latents=z_array,gts=gt_array)
            break
        if re.search(r'\.\d\.npz$',fpath):
            fpath = f'{fpath[:-5]}{int(fpath[-5])+1}.npz'
        else:
            fpath = fpath[:-4] + '.1.npz'


if __name__ == '__main__':
    utils.make_config()
    # Get current parameter configuration.
    config = fastargs.get_current_config()
    main(config)
