"""
Measure training time on Colorectal-YachidaS dataset.
"""
import sys
import argparse
import random
import statistics
import math
from datetime import datetime

sys.path.insert(0, "../")

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from microbiome_mvib.dataset import *
from microbiome_mvib.utils import *
from microbiome_mvib.downstream import classification
import timeit

# set seeds for reproducibility and make pytorch deterministic
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)

performance_metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']


def main(args):
    data_dir = os.path.join(args.root, 'data')

    svm_time = []
    rf_time = []
    rf_hpo_time = []

    dataset = MetabolicDataset(data_dir, 'Colorectal-YachidaS', 'cpu', scale=True, data=args.data)
    for i in range(args.repeat):
        # outer split
        train_ids, test_ids, y_train, y_test = dataset.train_test_split(0.2, i)

        if args.modality == 'M':
            X_train = dataset.markers[train_ids].numpy()
            X_test = dataset.markers[test_ids].numpy()
        elif args.modality == 'A':
            X_train = dataset.abundance[train_ids].numpy()
            X_test = dataset.abundance[test_ids].numpy()
        elif args.modality == 'AM':
            X_train = np.concatenate(
                (dataset.markers[train_ids].numpy(), dataset.abundance[train_ids].numpy()),
                axis=1
            )
            X_test = np.concatenate(
                (dataset.markers[test_ids].numpy(), dataset.abundance[test_ids].numpy()),
                axis=1
            )
        elif args.modality == 'AMM':
            X_train = np.concatenate(
                (dataset.markers[train_ids].numpy(), dataset.abundance[train_ids].numpy(), dataset.metabolite[train_ids].numpy()),
                axis=1
            )
            X_test = np.concatenate(
                (dataset.markers[test_ids].numpy(), dataset.abundance[test_ids].numpy(), dataset.metabolite[test_ids].numpy()),
                axis=1
            )

        #SVM
        clf = svm.SVC()
        start = timeit.default_timer()
        clf.fit(X=X_train, y=y_train)
        stop = timeit.default_timer()
        svm_time.append(stop - start)
        print('SVM | Modality {} | Time: '.format(args.modality), stop - start)

        #RF
        clf = RandomForestClassifier()
        start = timeit.default_timer()
        clf.fit(X=X_train, y=y_train)
        stop = timeit.default_timer()
        rf_time.append(stop - start)
        print('RF | Modality {} | Time: '.format(args.modality), stop - start)

        #RF-HPO
        start = timeit.default_timer()
        downstream_metrics, val_scores, test_scores = classification(X_train, X_test, y_train, y_test, args, ['rf'])
        stop = timeit.default_timer()
        rf_hpo_time.append(stop - start)
        print('RF-HPO | Modality {} | Time: '.format(args.modality), stop - start)

    print('SVM: {} +- {}'.format(statistics.mean(svm_time), statistics.stdev(svm_time)/args.repeat))
    print('RF: {} +- {}'.format(statistics.mean(rf_time), statistics.stdev(rf_time)/args.repeat))
    print('RF-HPO: {} +- {}'.format(statistics.mean(rf_hpo_time), statistics.stdev(rf_hpo_time)/args.repeat))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time complexity empirical benchmark')
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
    assert args.modality in ['M', 'A', 'AM', 'AMM']
    main(args)
