from sklearn import ensemble
import numpy as np
from dl_utils.label_funcs import discretize_labels


def compute_fairness(latents,gts):
  scores = {}
  factor_counts = [np.unique(gts[:,i]).shape[0] for i in range(gts.shape[1])]
  num_factors = len(factor_counts)

  # For each factor train a single predictive model.
  mean_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  max_fairness = np.zeros((num_factors, num_factors), dtype=np.float64)
  for i in range(num_factors):
    print(i)
    sample_size = 100
    b = np.concatenate([np.ones(sample_size),np.zeros(gts.shape[0]-sample_size)]).astype(bool)
    np.random.shuffle(b)
    model = ensemble.GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=3)
    model.fit(latents[b], gts[b, i])

    for j in range(num_factors):
      if i == j:
        continue
      counts = np.zeros((factor_counts[i], factor_counts[j]), dtype=np.int64)
      for c in np.unique(gts[:,j]):
        mask_for_this_value = gts[:,j]==c
        X = latents[mask_for_this_value]

        predictions = model.predict(X)
        counts[:, c] = np.bincount(predictions, minlength=factor_counts[i])
      mean_fairness[i, j], max_fairness[i, j] = inter_group_fairness(counts)

  scores.update(compute_scores_dict(mean_fairness, "mean_fairness"))
  #scores.update(compute_scores_dict(max_fairness, "max_fairness"))
  return scores


def compute_scores_dict(metric, prefix):
  """Computes scores for combinations of predictive and sensitive factors.

  Either average or take the maximum with respect to target and sensitive
  variable for all combinations of predictive and sensitive factors.

  Args:
    metric: Matrix of shape [num_factors, num_factors] with fairness scores.
    prefix: Prefix for the matrix in the returned dictionary.

  Returns:
    Dictionary containing all combinations of predictive and sensitive factors.
  """
  result = {}
  # Report min and max scores for each predictive and sensitive factor.
  for i in range(metric.shape[0]):
    for j in range(metric.shape[1]):
      if i != j:
        result["{}:pred{}:sens{}".format(prefix, i, j)] = metric[i, j]

  # Compute mean and max values across rows.
  rows_means = []
  rows_maxs = []
  for i in range(metric.shape[0]):
    relevant_scores = [metric[i, j] for j in range(metric.shape[1]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:pred{}:mean_sens".format(prefix, i)] = mean_score
    #result["{}:pred{}:max_sens".format(prefix, i)] = max_score
    rows_means.append(mean_score)
    rows_maxs.append(max_score)

  # Compute mean and max values across rows.
  column_means = []
  column_maxs = []
  for j in range(metric.shape[1]):
    relevant_scores = [metric[i, j] for i in range(metric.shape[0]) if i != j]
    mean_score = np.mean(relevant_scores)
    max_score = np.amax(relevant_scores)
    result["{}:sens{}:mean_pred".format(prefix, j)] = mean_score
    #result["{}:sens{}:max_pred".format(prefix, j)] = max_score
    column_means.append(mean_score)
    column_maxs.append(max_score)

  # Compute all combinations of scores.
  result["{}:mean_sens:mean_pred".format(prefix)] = np.mean(column_means)
  #result["{}:mean_sens:max_pred".format(prefix)] = np.mean(column_maxs)
  #result["{}:max_sens:mean_pred".format(prefix)] = np.amax(column_means)
  #result["{}:max_sens:max_pred".format(prefix)] = np.amax(column_maxs)
  #result["{}:mean_pred:mean_sens".format(prefix)] = np.mean(rows_means)
  #result["{}:mean_pred:max_sens".format(prefix)] = np.mean(rows_maxs)
  #result["{}:max_pred:mean_sens".format(prefix)] = np.amax(rows_means)
  #result["{}:max_pred:max_sens".format(prefix)] = np.amax(rows_maxs)

  return result


def inter_group_fairness(counts):
  """Computes the inter group fairness for predictions based on the TV distance.

  Args:
   counts: Numpy array with counts of predictions where rows correspond to
     predicted classes and columns to sensitive classes.

  Returns:
    Mean and maximum total variation distance of a sensitive class to the
      global average.
  """
  # Compute the distribution of predictions across all sensitive classes.
  overall_distribution = np.sum(counts, axis=1, dtype=np.float32)
  overall_distribution /= overall_distribution.sum()

  # Compute the distribution for each sensitive class.
  normalized_counts = np.array(counts, dtype=np.float32)
  counts_per_class = np.sum(counts, axis=0)
  normalized_counts /= np.expand_dims(counts_per_class, 0)

  # Compute the differences and sum up for each sensitive class.
  differences = normalized_counts - np.expand_dims(overall_distribution, 1)
  total_variation_distances = np.sum(np.abs(differences), 0) / 2.

  mean = (total_variation_distances * counts_per_class)
  mean /= counts_per_class.sum()

  return np.sum(mean), np.amax(total_variation_distances)

if __name__ == '__main__':
    correlation_info = 'single_1_01'
    data = np.load('../disentangled_clustering/experiments/3h{correlation_info}/3h{correlation_info}.npz')
    latents, gts = data['latents'], discretize_labels(data['gts'])
    #train_data = np.load('../disentangled_clustering/experiments/3dshapes_betaH_22-41/3dshapes_betaH_train_data_22-41.npz')
    #test_data = np.load('../disentangled_clustering/experiments/3dshapes_betaH_22-41/3dshapes_betaH_test_data_22-41.npz')
    #latents = np.concatenate((train_data['latents'],test_data['latents']),axis=0)
    #gts = np.concatenate((train_data['gts'],test_data['gts']),axis=0)


    from pprint import pprint
    results = compute_fairness(latents,gts)
    with open(f'3h{correlation_info}/3h{correlation_info}_full_fairness_results.json','w') as f:
        json.dump(results,f)
