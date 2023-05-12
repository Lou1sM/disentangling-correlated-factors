from fairness import compute_fairness
from vendi_score import vendi
import os
import numpy as np
from dl_utils.label_funcs import discretize_labels
import json
from sklearn.model_selection import train_test_split
from dci import _compute_dci


all_results = {}
#for latents_data_dir_name in ['3bsingle_1_01','3fsingle_1_01','3hdouble_2_04','3hsingle_2_01','3tsingle_1_01','mbsingle_1_01','chindependent']:
for latents_data_dir_name in os.listdir('hfs_results'):
    if not os.path.isdir(os.path.join('hfs_results',latents_data_dir_name)): continue
    if latents_data_dir_name.startswith('c'): continue
    if os.path.isfile(f'hfs_results/{latents_data_dir_name}/{latents_data_dir_name}_full_dci_results.json'):
        continue
    print(latents_data_dir_name)
    new_results = {}
    data_dir = os.path.join('hfs_results',latents_data_dir_name)
    data = np.load(f'{data_dir}/{latents_data_dir_name}.npz')
    latents, gts = data['latents'], discretize_labels(data['gts'])
    mutr,muts,ytr,yts = train_test_split(latents,gts)
    dci_results = _compute_dci(mutr.transpose(),ytr.transpose(),muts.transpose(),yts.transpose())
    #fairness_results = compute_fairness(latents,gts)
    #with open(f'{data_dir}/{latents_data_dir_name}_full_fairness_results.json','w') as f:
    #    json.dump(fairness_results,f)
    with open(f'{data_dir}/{latents_data_dir_name}_full_dci_results.json','w') as f:
        json.dump(dci_results,f)
    #new_results['fairness'] = fairness_results['mean_fairness:mean_sens:mean_pred']
    #new_results['vendi_diversity'] = vendi.score_dual(latents)
    new_results['dci-d'] = dci_results['disentanglement']['avg']
    new_results['dci-c'] = dci_results['completeness']['avg']
    new_results['dci-i'] = dci_results['informativeness']['avgts']
    all_results[latents_data_dir_name] = new_results

with open('hfs_results/dci_results.json','w') as f:
    json.dump(all_results,f)

