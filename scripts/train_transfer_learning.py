"""
Transfer learning experiments.
First, train MVIB on all diseases and compute performance metrics.
MVIB is pre-trained and initialized with the obtained weights, then fine-tuned.
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
from microbiome_mvib.saliency import Saliency

# set seeds for reproducibility and make pytorch deterministic
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_deterministic(True)

performance_metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']


def do_pre_training(pre_training_dataset, model, experiment_dir, seed):
    # do pre-training on all non-target diseases
    train_ids, val_ids, _, __ = pre_training_dataset.train_test_split(0.2, seed)

    # create train loader
    pre_training_train_sampler = SubsetRandomSampler(train_ids)
    pre_training_train_loader = torch.utils.data.DataLoader(
        pre_training_dataset,
        batch_size=args.batch_size,
        sampler=pre_training_train_sampler
    )

    # create val loader
    pre_training_val_sampler = SubsetRandomSampler(val_ids)
    pre_training_val_loader = torch.utils.data.DataLoader(
        pre_training_dataset,
        batch_size=args.batch_size,
        sampler=pre_training_val_sampler
    )

    # pre-training
    pre_trainer = Trainer(
        model=model,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        lambda_abundance=args.lambda_abundance,
        lambda_markers=args.lambda_markers,
        lambda_bce=args.lambda_bce,
        lambda_triplet=args.lambda_triplet,
        checkpoint_dir=experiment_dir,
        monitor='max'
    )
    print('Pre-training...')
    pre_train_state = pre_trainer.train(
        train_loader=pre_training_train_loader,
        val_loader=pre_training_val_loader,
        bce=True,
        triplet=True,
        autoencoding=False
    )
    print('End of pre-training.')
    return pre_train_state


def main(args):
    data_dir = os.path.join(args.root, 'data')
    result_dir = os.path.join(args.root, 'results')
    run_id = str(datetime.now().strftime("%d-%b-%Y--%H-%M-%S"))
    if not os.path.exists(os.path.join(result_dir, run_id)):
        os.mkdir(os.path.join(result_dir, run_id))

    # downstream models
    models = []
    if args.down:
        models = ['svm', 'rf']

    if int(args.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.gpu)

    overview = dict()  # a dictionary for results for all disease

    for disease in datasets_tl:
        disease_dir = os.path.join(result_dir, run_id, disease)
        if not os.path.exists(disease_dir):
            os.mkdir(disease_dir)

        val_mean = dict()
        test_mean = dict()
        test_std = dict()

        val_results = {model: [] for model in models}
        test_results = {model: [] for model in models}
        mvib_test_metrics = {metric: [] for metric in performance_metrics}

        pre_training_dataset = FullMicrobiomeDataset(
            data_dir, disease, device, scale=True, data=args.data, diseases=datasets_tl
        )
        dataset = MicrobiomeDataset(data_dir, disease, device, scale=True, data=args.data)
        saliency = Saliency(dataset)

        for i in range(args.repeat):
            experiment_dir = os.path.join(disease_dir, str(i))
            if not os.path.exists(experiment_dir) and not sys.gettrace():
                os.mkdir(experiment_dir)

            model = MVIB(
                n_latents=args.n_latents,
                abundance_dim=len(dataset[0][0]),
                marker_dim=len(dataset[0][1]),
                device=device
            )
            model = model.to(device)

            # do pre-training
            pre_train_state = do_pre_training(pre_training_dataset, model, experiment_dir, i)

            # outer split
            train_ids, test_ids, y_train, y_test = dataset.train_test_split(0.2, i)

            # inner split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)

            ensembler = Ensembler()
            mvib_val_metrics = []
            for fold, (inner_train_ids, val_ids) in enumerate(skf.split(train_ids, y_train)):
                if not args.ensemble:  # random split instead of kFold CV
                    inner_train_ids, val_ids, _, __ = model_selection.train_test_split(
                        train_ids,
                        y_train,
                        test_size=0.2,
                        random_state=i,
                        stratify=y_train
                    )
                else:
                    # Note: skf.split(X, y) returns indexes of the input array X
                    # we need the next 2 lines to get the node IDs from the array indexes returned by skf.split(X, y)
                    inner_train_ids = train_ids[inner_train_ids]
                    val_ids = train_ids[val_ids]

                # create train loader
                train_sampler = SubsetRandomSampler(inner_train_ids)
                train_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    sampler=train_sampler
                )

                # create val loader
                val_sampler = SubsetRandomSampler(val_ids)
                val_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    sampler=val_sampler
                )

                # instantiate MVIB with pre-training weights
                model, _ = Trainer.load_checkpoint(pre_train_state)
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
                    monitor=args.monitor,
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
                saliency.init()
                mu, logvar = model.infer(dataset.abundance, dataset.markers)
                prediction = model.classify(dataset.abundance[test_ids], dataset.markers[test_ids])
                ensembler.add(mu, prediction)
                saliency.update(
                    model.classify(dataset.abundance, dataset.markers),
                    dataset.labels_gpu
                )
                saliency.stop()

                if not args.ensemble:
                    break
                torch.cuda.empty_cache()

            # ensembling: average patient latent representations and class predictions
            mu, prediction = ensembler.avg()
            X_train = mu[train_ids]
            X_test = mu[test_ids]

            for metric in performance_metrics:
                if metric == 'roc_auc_score':
                    mvib_test_metrics[metric].append(getattr(metrics, metric)(y_test, prediction))
                    print('{} | Exp {} | MVIB Test {}: {}'.format(disease, i, metric,
                                                                  getattr(metrics, metric)(y_test, prediction)))
                else:
                    mvib_test_metrics[metric].append(getattr(metrics, metric)(y_test, prediction.round()))
                    print('{} | Exp {} | MVIB Test {}: {}'.format(disease, i, metric,
                                                                  getattr(metrics, metric)(y_test, prediction.round())))
            torch.cuda.empty_cache()

            # downstream classification
            downstream_metrics, val_scores, test_scores = classification(X_train, X_test, y_train, y_test, args, models)
            for model in models:
                val_results[model].append(downstream_metrics[model]['cross_val_best_score'])
                test_results[model].append(downstream_metrics[model]['test_roc_auc_score'])

            # add MVIB experiment scores to be saved in experiment results
            if args.bce or args.triplet:
                val_scores['MVIB-roc_auc_score'] = statistics.mean(mvib_val_metrics)
                for metric in performance_metrics:
                    test_scores['MVIB-'+metric] = mvib_test_metrics[metric][-1]

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

        if args.bce or args.triplet:
            # test results from ensemble of MVIB classification layers
            for metric in performance_metrics:
                test_mean['MVIB-'+metric] = statistics.mean(mvib_test_metrics[metric])
                test_std['MVIB-'+metric] = statistics.stdev(mvib_test_metrics[metric]) / math.sqrt(args.repeat)

        save_results(disease_dir, val_mean, test_mean, test_std, args)
        overview[disease] = {'val_mean': val_mean, 'test_mean': test_mean, 'test_std': test_std}
        if args.saliency:
            saliency.save(disease_dir)

    with open(os.path.join(result_dir, run_id, 'overview.json'), 'w') as fp:
        json.dump(overview, fp, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MVIB')
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
    parser.add_argument('--ensemble', dest='ensemble', action='store_true', help='use ensembling')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--down', dest='down', action='store_true', help='train downstream models')
    parser.set_defaults(down=False)
    parser.add_argument('--bce', dest='bce', action='store_true',
                        help='use binary cross-entropy on classification task for MVIB')
    parser.set_defaults(bce=False)
    parser.add_argument('--triplet', dest='triplet', action='store_true',
                        help='use triplet loss on classification task for MVIB')
    parser.set_defaults(triplet=False)
    parser.add_argument('--no-autoencoding', dest='autoencoding', action='store_false',
                        help='do not use autoencoding loss on classification task for MVIB')
    parser.set_defaults(autoencoding=True)
    parser.add_argument('--monitor', type=str, default='max', help='if max monitor val ROC AuC; if min val loss')
    parser.add_argument('--n-jobs', type=int, default=16,
                        help='sklearn.model_selection.GridSearchCV jobs; -1 to use all CPUs; -2 to leave one free')
    parser.add_argument('--data', type=str, default='default', help='dataset pre-processing')
    parser.add_argument('--saliency', dest='saliency', action='store_true', help='save saliency on disk')
    parser.set_defaults(saliency=False)

    args = parser.parse_args()
    print(args)
    assert args.repeat >= 1
    assert args.data in ['joint']

    main(args)
