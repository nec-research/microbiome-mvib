# coding=utf-8
#        Multimodal Variational Information Bottleneck
#
#   File:     train_metabolic.py
#   Authors:  Filippo Grazioli filippo.grazioli@neclab.eu
#             Roman Siarheyeu raman.siarheyeu@neclab.eu
#             Giampaolo Pileggi giampaolo.pileggi@neclab.eu
#             Andrea Meiser andrea.meiser@neclab.eu
#
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#
"""
Train MVIB on the Yachida et al. (PMID: 31171880) datasets with  3 modalities:
- species abundance
- strain maekers
- metabolites
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
from microbiome_mvib.metabolic_trainer import Trainer
from microbiome_mvib.utils import *

# set seeds for reproducibility and make pytorch deterministic
np.random.seed(7)
torch.manual_seed(7)
random.seed(7)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_deterministic(True)

performance_metrics = ['roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']


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

    overview = dict()  # a dictionary for results for all diseases

    for disease in ['Colorectal-YachidaS']:
        disease_dir = os.path.join(result_dir, run_id, disease)
        if not os.path.exists(disease_dir):
            os.mkdir(disease_dir)

        val_mean = dict()
        test_mean = dict()
        test_std = dict()

        val_results = {model: [] for model in models}
        test_results = {model: [] for model in models}
        mvib_test_metrics = {metric: [] for metric in performance_metrics}

        dataset = MetabolicDataset(data_dir, disease, device, scale=True, data=args.data)

        for i in range(args.repeat):
            experiment_dir = os.path.join(disease_dir, str(i))
            if not os.path.exists(experiment_dir) and not sys.gettrace():
                os.mkdir(experiment_dir)

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
                    device=device,
                    metabolite_dim=len((dataset[0][2]))
                )
                model = model.to(device)

                trainer = Trainer(
                    model=model,
                    epochs=args.epochs,
                    lr=args.lr,
                    beta=args.beta,
                    checkpoint_dir=experiment_dir,
                    monitor=args.monitor
                )

                # training
                state = trainer.train(train_loader, val_loader)
                model, val_best_roc_auc = trainer.load_checkpoint(state)
                mvib_val_metrics.append(val_best_roc_auc)
                del state

                model = model.to(device)
                model.eval()
                mu, logvar = model.infer(dataset.abundance, dataset.markers)
                prediction = model.classify(
                    dataset.abundance[test_ids],
                    dataset.markers[test_ids],
                    dataset.metabolite[test_ids]
                )
                ensembler.add(mu, prediction)
                save_latent(experiment_dir, mu, logvar, y_test, test_ids, fold)

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
            val_scores['MVIB-roc_auc_score'] = statistics.mean(mvib_val_metrics)
            for metric in performance_metrics:
                test_scores['MVIB-' + metric] = mvib_test_metrics[metric][-1]

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

        # test results from ensemble of MVIB classification layers
        for metric in performance_metrics:
            test_mean['MVIB-' + metric] = statistics.mean(mvib_test_metrics[metric])
            test_std['MVIB-' + metric] = statistics.stdev(mvib_test_metrics[metric]) / math.sqrt(args.repeat)

        save_results(disease_dir, val_mean, test_mean, test_std, args)
        overview[disease] = {'val_mean': val_mean, 'test_mean': test_mean, 'test_std': test_std}

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
    parser.add_argument('--n-latents', dest='n_latents', type=int, default=256, help='MVIB latent dimension')
    parser.add_argument('--bs', dest='batch_size', type=int, default=256, help='MVIB batch size')
    parser.add_argument('--root', type=str, default='/mnt/container-nle-microbiome/')
    parser.add_argument('--ensemble', dest='ensemble', action='store_true', help='use ensembling')
    parser.set_defaults(ensemble=False)
    parser.add_argument('--down', dest='down', action='store_true', help='train downstream models')
    parser.set_defaults(down=False)
    parser.add_argument('--monitor', type=str, default='max', help='if max monitor val ROC AuC; if min val loss')
    parser.add_argument('--n-jobs', type=int, default=16,
                        help='sklearn.model_selection.GridSearchCV jobs; -1 to use all CPUs; -2 to leave one free')
    parser.add_argument('--data', type=str, default='default', help='dataset pre-processing')

    args = parser.parse_args()
    print(args)
    assert args.repeat >= 1
    assert args.data in ['default']

    main(args)
