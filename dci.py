import numpy as np
import scipy
from sklearn import ensemble
import json


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, informativeness_dict = compute_importance_gbt(
            mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    scores["informativeness"] = informativeness_dict
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                                             dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        print(i)
        model = ensemble.GradientBoostingClassifier(n_estimators=3,n_iter_no_change=1,subsample=0.01,min_samples_leaf=40)
        #model = ensemble.HistGradientBoostingClassifier(max_iter=5)
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, {'tr':train_loss, 'ts':test_loss, 'avgtr': np.mean(train_loss), 'avgts': np.mean(test_loss)}


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                                                    base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return {'per_factor':  per_code.tolist(), 'avg': np.sum(per_code*code_importance)}


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return {'per_factor': per_factor.tolist(), 'avg': np.dot(per_factor,factor_importance)}


if __name__ == '__main__':
    mutr = np.transpose(np.load('train_data.npz')['latents'])
    ytr = np.transpose(np.load('train_data.npz')['gts'])
    muts = np.transpose(np.load('test_data.npz')['latents'])
    yts = np.transpose(np.load('test_data.npz')['gts'])
    c=_compute_dci(mutr,ytr,muts,yts)
    print(c)
    with open('test_sap.json','w') as f: json.dump(c,f)
