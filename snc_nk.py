import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch
import torch.nn.functional as F
from dl_utils.label_funcs import get_num_labels, get_trans_dict, get_trans_dict_from_cost_mat
from scipy import stats


def np_ent(p):
    counts = np.bincount(p)
    return stats.entropy(counts)

def normalize(x):
    return (x-x.min()) / (x.max() - x.min())

def cond_ent_for_alignment(x,y):
    num_classes = get_num_labels(y)
    #dividers_idx = np.arange(0,len(x),len(x)/num_classes).astype(int)
    dividers_idx = np.linspace(0,len(x) - len(x)/num_classes,num_classes).astype(int)
    bin_dividers = np.sort(x)[dividers_idx]
    bin_vals = sum([x<bd for bd in bin_dividers])
    total_ent = 0
    for bv in np.unique(bin_vals):
        bv_mask = bin_vals==bv
        gts_for_this_val = y[bv_mask]
        new_ent = np_ent(gts_for_this_val)
        total_ent += new_ent*bv_mask.sum()/len(x)
    return total_ent

class PredictorAligner():
    def __init__(self,latents_train, latents_test, labels_train, labels_test):
        self.results = {'train': {}, 'test': {}}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.N_train,self.nz = latents_train.shape
        self.N_test,self.n_labels = labels_test.shape
        assert latents_test.shape == (self.N_test, self.nz)
        assert labels_train.shape == (self.N_train, self.n_labels)

        self.latents_train = latents_train
        self.labels_train = labels_train
        self.latents_test = latents_test
        self.labels_test = labels_test
        latents = normalize(latents_train)
        self.nz = latents.shape[1]

        # Now align factors to neurons
        cost_mat_ = np.empty((self.nz,self.n_labels)) # tall thin matrix normally
        for i in range(self.nz):
            factor_latents = latents[:,i]
            for j in range(self.n_labels):
                factor_labels = labels_train[:,j]
                cost_mat_[i,j] = cond_ent_for_alignment(factor_latents,factor_labels)

        raw_ents = np.array([np_ent(self.labels_train[:,j]) for j in range(self.n_labels)])

        # cost_mat has i,j th entry as the cost of assigning factor
        # i to neuron j, computed as the negative mutual info
        cost_mat = np.transpose(cost_mat_ - raw_ents)

        # trans_dict relates factors to the distinct neurons select to
        # represent them, computed from cost_mat with Kuhn's algorithm
        self.trans_dict = get_trans_dict_from_cost_mat(cost_mat)

    def train_classif(self,x,y,x_test,y_test):
        ngts = max(y.max(), y_test.max()) + 1
        fc = nn.Sequential( nn.Linear(x.shape[1],256,device=self.device),
                            nn.ReLU(),
                            nn.Linear(256,ngts,device=self.device))
        opt = Adam(fc.parameters(),lr=1e-2) # training is faster with high lr + scheduler
        scheduler = lr_scheduler.ExponentialLR(opt,gamma=0.65)

        x_train_tensor = torch.tensor(x,device=self.device).float()
        y_train_tensor = torch.tensor(y,device=self.device)
        x_test_tensor = torch.tensor(x_test,device=self.device).float()
        y_test_tensor = torch.tensor(y_test,device=self.device)
        train_set = TensorDataset(x_train_tensor,y_train_tensor)
        test_set = TensorDataset(x_test_tensor,y_test_tensor)
        train_loader = DataLoader(train_set,shuffle=True,batch_size=4096)
        test_loader = DataLoader(test_set,shuffle=False,batch_size=4096)
        tol = 0
        best_acc = 0
        n_correct_train = 0
        n_correct_test = 0
        max_epochs = 75
        for i in range(max_epochs):
            if i > 0 and (i & (i-1) == 0): # power of 2
                scheduler.step()
            for batch_idx,(xb,yb) in enumerate(train_loader):
                preds = fc(xb.to(self.device))
                loss = F.cross_entropy(preds,yb.to(self.device))
                loss.backward(); opt.step(); opt.zero_grad()
                n_correct_train += (preds.argmax(axis=1)==yb).int().sum().item()

            train_acc = n_correct_train/self.N_train
            with torch.no_grad():
                for xb,yb in test_loader:
                    preds = fc(xb.to(self.device))
                    n_correct_test += (preds.argmax(axis=1)==yb).int().sum().item()
            test_acc = n_correct_train/self.N_train
            if train_acc > .99:
                break
            if train_acc>best_acc:
                tol = 0
                best_acc = train_acc
            else:
                tol += 1
                if tol == 4:
                    break
        return train_acc, test_acc

    def single_neuron_classification(self):
        train_results = []
        test_results = []
        for fn in range(self.n_labels):
            latent_to_exclude = self.trans_dict[fn]
            x = self.latents_train[:,latent_to_exclude]
            xt = self.latents_test[:,latent_to_exclude]
            y = self.labels_train[:,fn]
            yt = self.labels_test[:,fn]
            num_classes = get_num_labels(y)
            bin_dividers = np.sort(x)[np.arange(0,len(x),len(x)/num_classes).astype(int)]
            bin_dividers[0] = min(bin_dividers[0],min(xt))
            bin_vals = sum([x<bd for bd in bin_dividers]) # sum is over K np arrays where K is num classes, produces labels for the dset
            bin_vals_test = sum([xt<bd for bd in bin_dividers])
            trans_dict = get_trans_dict(bin_vals,y,subsample_size=30000)
            train_corrects = np.array([trans_dict[z] for z in bin_vals]) == y
            test_corrects = np.array([trans_dict[z] for z in bin_vals_test]) == yt
            train_chance_acc = ((np.bincount(self.labels_train[:,0])/self.N_train)**2).sum()
            test_chance_acc = ((np.bincount(self.labels_test[:,0])/self.N_test)**2).sum()
            train_snc = max(0, ((train_corrects.mean() - train_chance_acc) / (1-train_chance_acc)))
            test_snc = max(0, ((test_corrects.mean() - test_chance_acc) / (1-test_chance_acc)))
            train_results.append(train_snc)
            test_results.append(test_snc)
        self.results['train']['SNC'] = train_results
        self.results['test']['SNC'] = test_results

    def neuron_knockout(self):
        train_results = []
        test_results = []
        for fn in range(self.n_labels):
            latent_to_exclude = self.trans_dict[fn]
            X = np.delete(self.latents_train,latent_to_exclude,1)
            X_test = np.delete(self.latents_test,latent_to_exclude,1)
            y = self.labels_train[:,fn]
            y_test = self.labels_test[:,fn]
            train_acc, test_acc = self.train_classif(self.latents_train,y,self.latents_test,y_test)
            nk_train_acc, nk_test_acc = self.train_classif(X,y,X_test,y_test)
            train_results.append(train_acc-nk_train_acc)
            test_results.append(test_acc-nk_test_acc)
        self.results['train']['NK'] = train_results
        self.results['test']['NK'] = test_results

def snc_nk(latents_train, labels_train, latents_test=None, labels_test=None):
    """Both metrics are computed together because they use the same alignment of neurons to factors."""
    assert (latents_test is None) == (labels_test is None)
    if latents_test is None: # split off random 25% as test set
        test_idxs = np.random.choice(np.arange(len(latents_train)),len(latents_train)//4,replace=False)
        latents_test = latents_train[test_idxs]
        labels_test = labels_train[test_idxs]
        latents_train = np.delete(latents_train,test_idxs,0)
        labels_train = np.delete(labels_train,test_idxs,0)
    pa = PredictorAligner(latents_train, latents_test, labels_train, labels_test)
    pa.single_neuron_classification()
    #pa.neuron_knockout()
    return pa.results
