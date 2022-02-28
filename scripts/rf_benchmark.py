"""
Train Random Forest on all diseases and compute performance metrics.
"""
import sys
import argparse
import random
import statistics
import math
from datetime import datetime

sys.path.insert(0, "../")

from microbiome_mvib.downstream import classification
from microbiome_mvib.dataset import *
from microbiome_mvib.utils import *

# set seeds for reproducibility and make pytorch deterministic
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)

performance_metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']


def main(args):
    data_dir = os.path.join(args.root, 'data')
    result_dir = os.path.join(args.root, 'results')
    run_id = str(datetime.now().strftime("%d-%b-%Y--%H-%M-%S"))
    if not os.path.exists(os.path.join(result_dir, run_id)):
        os.mkdir(os.path.join(result_dir, run_id))

    # classifier models
    models = ['rf']

    overview = dict()  # a dictionary for results for all diseases

    diseases = datasets

    for disease in diseases:
        disease_dir = os.path.join(result_dir, run_id, disease)
        if not os.path.exists(disease_dir):
            os.mkdir(disease_dir)

        val_mean = dict()
        test_mean = dict()
        test_std = dict()

        val_results = {model: [] for model in models}
        test_results = {model: [] for model in models}

        dataset = MicrobiomeDataset(data_dir, disease, 'cpu', scale=True, data=args.data)

        for i in range(args.repeat):
            experiment_dir = os.path.join(disease_dir, str(i))
            if not os.path.exists(experiment_dir) and not sys.gettrace():
                os.mkdir(experiment_dir)

            # outer split
            train_ids, test_ids, y_train, y_test = dataset.train_test_split(0.2, i)

            if args.modality == 'markers':
                X_train = dataset.markers[train_ids].numpy()
                X_test = dataset.markers[test_ids].numpy()
            elif args.modality == 'abundance':
                X_train = dataset.abundance[train_ids].numpy()
                X_test = dataset.abundance[test_ids].numpy()
            elif args.modality == 'both':
                X_train = np.concatenate(
                    (dataset.markers[train_ids].numpy(), dataset.abundance[train_ids].numpy()),
                    axis=1
                )
                X_test = np.concatenate(
                    (dataset.markers[test_ids].numpy(), dataset.abundance[test_ids].numpy()),
                    axis=1
                )

            downstream_metrics, val_scores, test_scores = classification(X_train, X_test, y_train, y_test, args, models)

            for model in models:
                val_results[model].append(downstream_metrics[model]['cross_val_best_score'])
                test_results[model].append(downstream_metrics[model]['test_roc_auc_score'])

            save_experiment(
                downstream_metrics,
                val_scores,
                test_scores,
                disease,
                result_dir,
                run_id,
                str(i)
            )

        # results from downstream classifiers
        for model in models:
            val_mean[model] = statistics.mean(val_results[model])
            test_mean[model] = statistics.mean(test_results[model])
            test_std[model] = statistics.stdev(test_results[model]) / math.sqrt(args.repeat)

        save_results(disease_dir, val_mean, test_mean, test_std, args)
        overview[disease] = {'val_mean': val_mean, 'test_mean': test_mean, 'test_std': test_std}

    with open(os.path.join(result_dir, run_id, 'overview.json'), 'w') as fp:
        json.dump(overview, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RF')
    parser.add_argument('--repeat', type=int, default=5, help='how many time experiments are repeated')
    parser.add_argument('--root', type=str, default='/mnt/container-nle-microbiome/')
    parser.add_argument('--data', type=str, default='default', help='dataset pre-processing')
    parser.add_argument('--modality', type=str, default='markers', help='which modalities to consdier')
    parser.add_argument('--n-jobs', type=int, default=16,
                        help='sklearn.model_selection.GridSearchCV jobs; -1 to use all CPUs; -2 to leave one free')
    args = parser.parse_args()
    print(args)
    assert args.repeat >= 1
    assert args.data in ['default', 'joint']
    assert args.modality in ['markers', 'abundance', 'both']
    main(args)
