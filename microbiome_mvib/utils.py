"""
Utilities.
"""
import os
import json
import numpy as np


def save_results(disease_dir, val_mean, test_mean, test_std, args):
    """
    Save the overall results for all diseases and all experiments.
    """
    results = {
        'val_mean': val_mean,
        'test_mean': test_mean,
        'test_std': test_std,
    }
    results = {**results, **vars(args)}

    with open(os.path.join(disease_dir, 'overall_results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


def save_experiment(downstream_metrics, val_mean, test_mean, disease, result_dir, run_id, tuning_id):
    """
    Save the the results of one experiment for one disease.
    """
    result_dir = os.path.join(result_dir, run_id, disease, tuning_id)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    downstream_metrics['val'] = val_mean
    downstream_metrics['test'] = test_mean
    with open(os.path.join(result_dir, 'experiment.json'), 'w') as fp:
        json.dump(downstream_metrics, fp, indent=4)


def save_latent(experiment_dir, mu, logvar, y_test, test_ids, fold):
    """
    Save latent space embeddings and labels.
    """
    np.save(os.path.join(experiment_dir, 'mu_{}'.format(fold)), mu[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'logvar_{}'.format(fold)), logvar[test_ids].cpu().detach().numpy())
    np.save(os.path.join(experiment_dir, 'label_{}'.format(fold)), y_test)
