import os
from fairness import compute_fairness
import json
from sklearn.model_selection import train_test_split
import numpy as np
from vendi_score import vendi
from ignite.engine import Engine
from ignite.metrics import FID
import argparse
import torch
import datasets
import dent
import utils
from dci import _compute_dci
from dl_utils.label_funcs import discretize_labels
from dl_utils.tensor_funcs import numpyify, cudify
from snc_nk import snc_nk


parser = argparse.ArgumentParser()
parser.add_argument('--chkpt_dir', '-c', type=str, default='.')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
ARGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
device = torch.device('cuda')

chkpt_path = os.path.join(ARGS.chkpt_dir, 'chkpt.pth.tar')
chkpt = torch.load(chkpt_path)
metadata = chkpt['metadata']
config = utils.overwrite_config(metadata)

all_results = {}

def eval_step(engine, batch):
    return batch


default_evaluator = Engine(eval_step)

feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
feature_extractor.cuda()
for cor_setting in ['no_corr','single_1_01','double_1_01','conf_1_02']:
#for cor_setting in ['no_corr','single_1_01']:
    results_cor = {}
    print(f'\nUSING DATASET WITH CORRELATION SETTING: {cor_setting.upper()}')
    if cor_setting == 'no_corr':
        correlations_filepath = 'constraints/avail_correlations.yaml:base'
    else:
        correlations_filepath = f'constraints/avail_correlations.yaml:{metadata["data.name"]}_{cor_setting}'

    constraints_filepath = config['constraints.file']
    if constraints_filepath == 'none':
        constraints_filepath = None
    correlations_filepath = config['constraints.correlations_file']
    if correlations_filepath == 'none':
        correlations_filepath = None

    train_loader, _ = datasets.get_dataloaders(
        shuffle=True,
        device=device,
        batch_size=ARGS.batch_size,
        return_pairs='pairs' in config['train.supervision'],
        root=config['data.root'],
        k_range=config['data.k_range'],
        constraints_filepath=constraints_filepath,
        correlations_filepath=correlations_filepath)

    model = dent.model_select(device, name=config['model.name'], img_size=config['data.img_size'])
    model.load_state_dict(chkpt['model'])
    _ = model.to(device)
    _ = model.eval()
    X_list = []
    z_list = []
    gt_list = []
    rec_list = []
    noisy_rec_list = []
    samples_list = []
    traversed_rec_list = []
    t_mask = torch.cat([torch.tensor([[-3],[-2],[-1],[1],[2],[3]])*(torch.arange(10)==i) for i in range(10)]).cuda()
    #X = torch.tensor(train_loader.dataset.imgs/255).permute(0,3,1,2).float()
    X = torch.rand(1000,3,64,64)
    for k in ['recs','noisy_recs','samples','traversals']:
        data_list = []
        results_cor[k] = {}
        print(f'\tGenerating {k}')
        for i, (xb,yb) in enumerate(train_loader):
            if i%100 == 0:
                print(i)
            if k == 'recs':
                data_list.append(model(xb.cuda())['reconstructions'].detach().cpu())
            elif k == 'noisy_recs':
                z = model.encoder(xb.cuda())
                mean,logvar = z['stats_qzx'].unbind(2)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                noisy_rec_ = model.decoder(mean + std*eps)['reconstructions']
                data_list.append(noisy_rec_.detach().cpu())
            elif k == 'samples':
                #z = model(xb.cuda())
                #samples_ = model.decoder(torch.randn_like(mean))['reconstructions']
                samples_ = model.decoder(torch.randn(ARGS.batch_size,10).cuda())['reconstructions']
                data_list.append(samples_.detach().cpu())
            elif k == 'traversals':
                z = model.encoder(xb.cuda())
                mean,logvar = z['stats_qzx'].unbind(2)
                std = torch.exp(0.5 * logvar)
                traversed_mean = t_mask*std.unsqueeze(1) + mean.unsqueeze(1)
                # check exactly one neuron has been changed in each row
                assert ((traversed_mean==mean.unsqueeze(1)).int().sum(axis=2)==9).all()
                traversed_mean = traversed_mean.view(-1,10)
                #traversed_rec_ = model.decoder(traversed_mean)['reconstructions']
                traversed_rec_ = torch.cat([model.decoder(traversed_mean[i*1000:int((i+1)*1000)])['reconstructions'] for i in range(int(len(traversed_mean)/1000)+1)])
                #traversed_rec_list.append(traversed_rec_.detach().cpu())
                data_list.append(traversed_rec_.detach().cpu())
                if i*ARGS.batch_size >= len(X)/60:
                    break
            break

        v = torch.cat(data_list)
        del data_list

        print('\tComputing Vendi and FID scores')

        #feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True, weights=weights=ResNet18_Weights.DEFAULT)
        #sim_func = lambda a, b: np.exp(-np.abs(a - b))
        n = min(10000,len(X),len(v))
        fid = FID(feature_extractor=feature_extractor,num_features=1000,device='cuda')
        fid.attach(default_evaluator, "fid")
        results_cor[k]['fid'] = default_evaluator.run([[v[:n],X[:n]]]).metrics['fid']
        embedded_for_vendi = np.concatenate([numpyify(feature_extractor(cudify(v[i*1000:(i+1)*1000]))) for i in range(int(len(X)/1000 + 1))])
        #results_cor[k]['vendi'] = embedding_vendi_score(v, device="cuda")
        results_cor[k]['vendi'] = float(vendi.score_dual(embedded_for_vendi)) #so JSON-serializable

        del fid
        del v

        data_list = []
        gt_list = []
    print('Generating for DCI and SNC')
    for i, (xb,yb) in enumerate(train_loader):
        z = model(xb.cuda())
        mean,logvar = z['stats_qzx'].unbind(2)
        data_list.append(numpyify(mean))
        gt_list.append(numpyify(yb))
        if i == 15:
            break
    latents = np.concatenate(data_list,axis=0)
    gts = discretize_labels(np.concatenate(gt_list,axis=0))
    del data_list

    print('Computing DCI...')
    mutr,muts,ytr,yts = train_test_split(latents,gts)
    dci_results = _compute_dci(mutr.transpose(),ytr.transpose(),muts.transpose(),yts.transpose())
    results_cor['dci-d-avg'] = dci_results['disentanglement']['avg']
    results_cor['dci-c-avg'] = dci_results['completeness']['avg']
    results_cor['dci-i-avg'] = dci_results['informativeness']['avgts']
    results_cor['dci-full'] = dci_results

    print('Computing SNC...')
    snc_nk_results = snc_nk(mutr,ytr,muts,yts)
    results_cor['snc'] = np.mean(snc_nk_results['test']['SNC'])
    #results_cor['nk'] = np.mean(snc_nk_results['test']['NK'])
    results_cor['snc-nk-full'] = snc_nk_results

    print('Computing fairness...')
    fairness_results = compute_fairness(latents,gts)
    results_cor['fairness'] = fairness_results['mean_fairness:mean_sens:mean_pred']
    results_cor['fairness-full'] = fairness_results

    all_results[cor_setting] = results_cor


import pdb; pdb.set_trace()  # XXX BREAKPOINT
with open(os.path.join(ARGS.chkpt_dir,'metrics.json'), 'w') as f:
    json.dump(all_results,f)
