"""
Cross-study generalization experiments.
Train MVIB on a source dataset and test it on a different one.
"""
import sys
import argparse
import random
import statistics
import math
from datetime import datetime
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

sys.path.insert(0, "../")

from microbiome_mvib.downstream import classification
from microbiome_mvib.dataset import *
from microbiome_mvib.mvib import MVIB, Ensembler
from microbiome_mvib.trainer import Trainer
from microbiome_mvib.utils import *

# set seeds for reproducibility and make pytorch deterministic
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_deterministic(True)

performance_metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']

source_target_pairs = (
    ('WT2D', 'T2D'),
    ('T2D', 'WT2D'),
    ('Delta-Obesity', 'Obesity'),
    ('Delta-Colorectal', 'Colorectal'),
    ('Obesity', 'Delta-Obesity'),
    ('Colorectal', 'Delta-Colorectal')
)


def main(args):
    data_dir = os.path.join(args.root, 'data')
    result_dir = os.path.join(args.root, 'results')
    run_id = str(datetime.now().strftime("%d-%b-%Y--%H-%M-%S"))
    if not os.path.exists(os.path.join(result_dir, run_id)):
        os.mkdir(os.path.join(result_dir, run_id))

    if int(args.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.gpu)

    overview = dict()  # a dictionary for results for all diseases

    for (source, target) in source_target_pairs:
        disease_dir = os.path.join(result_dir, run_id, source+'_'+target)
        if not os.path.exists(disease_dir):
            os.mkdir(disease_dir)

        val_mean = {'RF': dict(), 'MVIB': dict()}
        test_mean = {'RF': dict(), 'MVIB': dict()}
        test_std = {'RF': dict(), 'MVIB': dict()}

        mvib_val_metrics = []
        mvib_test_metrics = {metric: [] for metric in performance_metrics}
        rf_val_metrics = []
        rf_test_metrics = {metric: [] for metric in performance_metrics}

        dataset = MicrobiomeDataset(data_dir, source, device, scale=True, data='joint')
        target_dataset = MicrobiomeDataset(data_dir, target, device, scale=True, data='joint')
        ensembler = Ensembler()

        for i in range(args.repeat):
            train_ids, val_ids, _, __ = dataset.train_test_split(0.2, i)
            experiment_dir = os.path.join(disease_dir, str(i))
            if not os.path.exists(experiment_dir) and not sys.gettrace():
                os.mkdir(experiment_dir)

            # create train loader
            train_sampler = SubsetRandomSampler(train_ids)
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
            )

            # create val loader
            val_sampler = SubsetRandomSampler(val_ids)
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=val_sampler,
            )

            model = MVIB(
                n_latents=args.n_latents,
                abundance_dim=len(dataset[0][0]),
                marker_dim=len(dataset[0][1]),
                device=device
            )
            model = model.to(device)

            trainer = Trainer(
                model=model,
                epochs=args.epochs,
                lr=args.lr,
                beta=args.beta,
                lambda_abundance=args.lambda_abundance,
                lambda_markers=args.lambda_markers,
                lambda_bce=args.lambda_bce,
                lambda_triplet=args.lambda_triplet,
                checkpoint_dir=experiment_dir,
                monitor=args.monitor
            )

            # training
            state = trainer.train(
                train_loader,
                val_loader,
                args.bce,
                args.triplet,
                args.autoencoding
            )
            model, val_best_roc_auc = trainer.load_checkpoint(state)
            mvib_val_metrics.append(val_best_roc_auc)
            del state

            model = model.to(device)
            model.eval()
            mu, logvar = model.infer(target_dataset.abundance, target_dataset.markers)
            prediction = model.classify(target_dataset.abundance, target_dataset.markers)
            ensembler.add(mu, prediction)

            torch.cuda.empty_cache()

            # random forest benchmark (repeated 5 times because non-deterministic)
            abundance_train = dataset.abundance.cpu().numpy()
            markers_train = dataset.markers.cpu().numpy()
            X_train = np.concatenate((abundance_train, markers_train), axis=1)
            abundance_test = target_dataset.abundance.cpu().numpy()
            markers_test = target_dataset.markers.cpu().numpy()
            X_test = np.concatenate((abundance_test, markers_test), axis=1)

            y_train = dataset.labels
            y_test = target_dataset.labels

            downstream_metrics, _, __ = classification(X_train, X_test, y_train, y_test, args, ['rf'])

            rf_val_metrics.append(downstream_metrics['rf']['cross_val_best_score'])
            for metric in performance_metrics:
                rf_test_metrics[metric].append(downstream_metrics['rf']['test_' + metric])

        mu, prediction = ensembler.avg()

        for metric in performance_metrics:
            if metric == 'roc_auc_score':
                m = getattr(metrics, metric)(target_dataset.labels, prediction)
                mvib_test_metrics[metric].append(m)
                print('{} | MVIB Test {}: {}'.format(source+'_'+target, metric, m))
            else:
                m = getattr(metrics, metric)(target_dataset.labels, prediction.round())
                mvib_test_metrics[metric].append(m)
                print('{} | MVIB Test {}: {}'.format(source+'_'+target, metric, m))
        torch.cuda.empty_cache()

        # test and val results for MVIB and RF
        for metric in performance_metrics:
            test_mean['MVIB'][metric] = statistics.mean(mvib_test_metrics[metric])
            test_mean['RF'][metric] = statistics.mean((rf_test_metrics[metric]))

            # the std is computed only for RF
            # the pytorch implementation of MVIB is deterministic, so the experiment is repeated only once
            # conversely, the sklearn implementation of RF is non-deterministic, so we repeat the experiment 5 times
            test_std['RF'][metric] = statistics.stdev(rf_test_metrics[metric]) / math.sqrt(args.repeat)

        val_mean['MVIB']['roc_auc_score'] = statistics.mean(mvib_val_metrics)
        val_mean['RF']['roc_auc_score'] = statistics.mean(rf_val_metrics)

        save_results(disease_dir, val_mean, test_mean, test_std, args)
        overview[source+'_'+target] = {'val_mean': val_mean, 'test_mean': test_mean, 'test_std': test_std}

    with open(os.path.join(result_dir, run_id, 'overview.json'), 'w') as fp:
        json.dump(overview, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cross-study generalization')
    parser.add_argument('--gpu', type=str, default='0', help='which GPU to use; -1 for CPU-only')
    parser.add_argument('--repeat', type=int, default=5, help='how many time experiments are repeated')
    parser.add_argument('--epochs', type=int, default=200, help='MVIB training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--beta', dest='beta', type=float, default=1,
                        help='multiplier for KL divergence')
    parser.add_argument('--lambda-abundance', type=float, default=1.,
                        help='multiplier for abundance reconstruction')
    parser.add_argument('--lambda-markers', type=float, default=1.,
                        help='multiplier for markers reconstruction')
    parser.add_argument('--lambda-bce', type=float, default=1.,
                        help='multiplier for bce classification loss')
    parser.add_argument('--lambda-triplet', type=float, default=1.,
                        help='multipilier for triplet loss')
    parser.add_argument('--n-latents', dest='n_latents', type=int, default=256, help='MVIB latent dimension')
    parser.add_argument('--bs', dest='batch_size', type=int, default=256, help='MVIB batch size')
    parser.add_argument('--root', type=str, default='/mnt/container-nle-microbiome/')
    parser.add_argument('--bce', dest='bce', action='store_true',
                        help='use binary cross-entropy on classification task for MVIB')
    parser.set_defaults(bce=True)
    parser.add_argument('--triplet', dest='triplet', action='store_true', help='use triplet loss for MVIB')
    parser.set_defaults(triplet=False)
    parser.add_argument('--no-autoencoding', dest='autoencoding', action='store_false', help='do autoencoding for MVIB')
    parser.set_defaults(autoencoding=True)
    parser.add_argument('--monitor', type=str, default='max', help='if max monitor val ROC AuC; if min val loss')
    parser.add_argument('--n-jobs', type=int, default=16,
                        help='sklearn.model_selection.GridSearchCV jobs; -1 to use all CPUs; -2 to leave one free')

    args = parser.parse_args()
    print(args)
    assert args.repeat >= 1
    assert args.bce or args.triplet, 'Either BCE or triplet loss is expected'

    main(args)
