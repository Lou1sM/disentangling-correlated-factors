from fairness import compute_fairness
from vendi_score import vendi
import os
import numpy as np
from dl_utils.label_funcs import discretize_labels
import json


all_results = {}
for latents_data_dir_name in os.listdir('hfs_results'):
    if not os.path.isdir(os.path.join('hfs_results',latents_data_dir_name)): continue
    correlation_info = latents_data_dir_name[2:]
    print(correlation_info)
    # following naming convention in constraints/avail_correlations.yml
    new_results = {}
    data_dir = os.path.join('hfs_results',latents_data_dir_name)
    data = np.load(f'{data_dir}/3h{correlation_info}.npz')
    latents, gts = data['latents'], discretize_labels(data['gts'])
    fairness_results = compute_fairness(latents,gts)
    with open(f'{data_dir}/3h{correlation_info}_full_fairness_results.json','w') as f:
        json.dump(fairness_results,f)
    new_results['fairness'] = fairness_results['mean_fairness:mean_sens:mean_pred']
    new_results['vendi_diversity'] = vendi.score_dual(latents)
    all_results[correlation_info] = new_results

with open('hfs_results/all_results.json','w') as f:
    json.dump(all_results,f)

