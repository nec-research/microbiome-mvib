# coding=utf-8
#        Multimodal Variational Information Bottleneck
#
#   File:     cross_study.py
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
