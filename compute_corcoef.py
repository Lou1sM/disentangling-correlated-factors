import numpy as np
import json

with open('hfs_results/3dshapes_results.json') as f:
    d = json.load(f)

with open('hfs_results/mpi3d_results.json') as f:
    d.update(json.load(f))

with open('hfs_results/all_results.json') as f:
    d.update(json.load(f))

#with open('hfs_results/dci_results.json') as f:
    #dci_d = json.load(f)
    #for k,v in d.items():
        #v.update(dci_d[k])

for k,v in d.items():
    with open(f'hfs_results/{k}/{k}_full_dci_results.json') as f:
        dci = json.load(f)
        v['dci-d'] = dci['disentanglement']['avg']

breakpoint()
#fairness_vs_vendi_mat = np.array([list(x.values()) for x in d.values()])
fairness_vs_vendi_vs_dci_mat = np.array([[x['fairness'],x['vendi_diversity'],x['dci-d']] for x in d.values()])

with open('hfs_results/all_results.json','w') as f:
    json.dump(d,f)

print(np.corrcoef(np.transpose(fairness_vs_vendi_vs_dci_mat)))
