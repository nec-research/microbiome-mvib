# coding=utf-8
#        Multimodal Variational Information Bottleneck
#
#   File:     trainer.py
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
Trainer class.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mvib.mvib import MVIB


class Trainer:
    """
    A class to automate training.
    It stores the weights of the model at the epoch where a validation score is maximized.
    """
    def __init__(
            self, model, epochs, lr, beta,
            lambda_abundance, lambda_markers,
            lambda_bce, lambda_triplet, checkpoint_dir, monitor
    ):
        self.model = model
        self.epochs = epochs
        self.beta = beta
        self.lambda_abundance = lambda_abundance
        self.lambda_markers = lambda_markers
        self.lambda_bce = lambda_bce
        self.lambda_triplet = lambda_triplet
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.logits_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.triplet_loss = torch.nn.TripletMarginLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, monitor, patience=10, cooldown=10, factor=0.5)

    def train(self, train_loader, val_loader, bce, triplet, autoencoding):
        """
        The main training loop.
        """
        if self.monitor == 'max':
            best_val_score = 0
        else:
            best_val_score = float('inf')
        is_best = False
        state = None

        for epoch in range(1, self.epochs + 1):
            self.train_step(train_loader, bce, triplet, autoencoding)
            val_loss, val_roc_auc = self.evaluate(val_loader, bce, triplet, autoencoding)

            if self.monitor == 'max':
                self.scheduler.step(val_roc_auc)
                is_best = val_roc_auc > best_val_score
                best_val_score = max(val_roc_auc, best_val_score)
            elif self.monitor == 'min':
                self.scheduler.step(val_loss)
                is_best = val_loss < best_val_score
                best_val_score = min(val_loss, best_val_score)

            if is_best:
                state = {
                    'state_dict': self.model.state_dict(),
                    'best_val_score': best_val_score,
                    'n_latents': self.model.n_latents,
                    'abundance_dim': self.model.abundance_dim,
                    'marker_dim': self.model.marker_dim,
                    'device': self.model.device,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch
                }
                print('====> Epoch: {} | Val score: {:.4f} | Best model'.format(state['epoch'], best_val_score))

        return state

    def train_step(self, train_loader, bce, triplet, autoencoding):
        """
        Compute loss and do backpropagation for a single epoch.
        """
        self.model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (abundance, markers, gt, abundance_gt, markers_gt) in enumerate(train_loader):
            abundance = Variable(abundance)
            markers = Variable(markers)
            batch_size = len(abundance)

            # refresh the optimizer
            self.optimizer.zero_grad()

            # pass data through model
            recon_abundance_1, recon_markers_1, mu_1, logvar_1, classification_logits_1 = self.model(abundance, markers)
            recon_abundance_2, recon_markers_2, mu_2, logvar_2, classification_logits_2 = self.model(abundance)
            recon_abundance_3, recon_markers_3, mu_3, logvar_3, classification_logits_3 = self.model(markers=markers)

            # compute ELBO/KL divergence for each data combination (sub-sampled training paradigm)
            joint_loss = self.elbo_KL_loss(
                recon_abundance_1, abundance_gt,
                recon_markers_1, markers_gt,
                mu_1, logvar_1,
                self.lambda_abundance, self.lambda_markers, self.beta, autoencoding
            )

            abundance_loss = self.elbo_KL_loss(
                recon_abundance_2, abundance_gt,
                None, None,
                mu_2, logvar_2,
                self.lambda_abundance, self.lambda_markers, self.beta, autoencoding
            )

            markers_loss = self.elbo_KL_loss(
                None, None,
                recon_markers_3, markers_gt,
                mu_3, logvar_3,
                self.lambda_abundance, self.lambda_markers, self.beta, autoencoding
            )

            train_loss = joint_loss + abundance_loss + markers_loss

            # compute binary cross entropy for each data combination (sub-sampled training paradigm)
            if bce:
                classification_loss = self.logits_bce(classification_logits_1, gt) + \
                                      self.logits_bce(classification_logits_2, gt) + \
                                      self.logits_bce(classification_logits_3, gt)
                classification_loss = torch.mean(classification_loss)
                train_loss = train_loss + self.lambda_bce * classification_loss

            # compute triplet loss for each data combination (sub-sampled training paradigm)
            if triplet:
                triplet_loss = self.get_triplet_loss(mu_1, mu_2, mu_3, gt)
                train_loss = train_loss + self.lambda_triplet * triplet_loss

            train_loss_meter.update(train_loss.item(), batch_size)

            # compute gradients and take step
            train_loss.backward()
            self.optimizer.step()

    def evaluate(self, val_loader, bce, triplet, autoencoding):
        """
        Evaluate model performance on validation (or test) set.
        """
        self.model.eval()
        val_loss_meter = AverageMeter()
        val_roc_auc = AverageMeter()

        with torch.no_grad():
            for batch_idx, (abundance, markers, gt, abundance_gt, markers_gt) in enumerate(val_loader):

                abundance = Variable(abundance)
                markers = Variable(markers)
                batch_size = len(abundance)

                recon_abundance_1, recon_markers_1, mu_1, logvar_1, classification_logits_1 = self.model(abundance, markers)
                recon_abundance_2, recon_markers_2, mu_2, logvar_2, classification_logits_2 = self.model(abundance)
                recon_abundance_3, recon_markers_3, mu_3, logvar_3, classification_logits_3 = self.model(markers=markers)

                # compute ELBO/KL divergence for each data combination (sub-sampled training paradigm)
                joint_loss = self.elbo_KL_loss(
                    recon_abundance_1, abundance_gt,
                    recon_markers_1, markers_gt,
                    mu_1, logvar_1, autoencoding
                )

                abundance_loss = self.elbo_KL_loss(
                    recon_abundance_2, abundance_gt,
                    None, None,
                    mu_2, logvar_2, autoencoding
                )

                markers_loss = self.elbo_KL_loss(
                    None, None,
                    recon_markers_3, markers_gt,
                    mu_3, logvar_3, autoencoding
                )

                val_loss = joint_loss + abundance_loss + markers_loss

                # compute binary cross entropy for each data combination (sub-sampled training paradigm)
                if bce:
                    classification_loss = self.logits_bce(classification_logits_1, gt) + \
                                          self.logits_bce(classification_logits_2, gt) + \
                                          self.logits_bce(classification_logits_3, gt)
                    classification_loss = torch.mean(classification_loss)
                    val_loss = val_loss + self.lambda_bce * classification_loss

                # compute triplet loss for each data combination (sub-sampled training paradigm)
                if triplet:
                    triplet_loss = self.get_triplet_loss(mu_1, mu_2, mu_3, gt)
                    val_loss = val_loss + self.lambda_triplet * triplet_loss

                val_loss_meter.update(val_loss.item(), batch_size)

                if bce or triplet:
                    val_roc_auc.update(roc_auc_score(
                        gt.cpu().numpy().squeeze().astype('int'),
                        np.nan_to_num(torch.sigmoid(classification_logits_1).detach().cpu().numpy().squeeze())
                    ), batch_size)

        return val_loss_meter.avg, val_roc_auc.avg

    def elbo_KL_loss(self, recon_1, x_1, recon_2, x_2, mu, logvar,
                  lambda_1=1.0, lambda_2=1.0, beta=1, autoencoding=True):
        """
        This method computes the ELBO loss if the `autoencoding` parameter is True.
        See: https://papers.nips.cc/paper/2019/file/0ae775a8cb3b499ad1fca944e6f5c836-Paper.pdf

        If the `autoencoding` parameter is False, the method only returns the KL divergence.
        """
        loss_1, loss_2 = 0, 0  # default params

        # Gaussian MLP decoder: log likelihood loss for a N(mu, var)
        if recon_1 is not None and x_1 is not None:
            mu_1, logvar_1 = recon_1
            loss_1 = -torch.sum(
                (-0.5 * np.log(2.0 * np.pi))
                + (-0.5 * logvar_1)
                + ((-0.5 / torch.exp(logvar_1)) * (x_1 - mu_1) ** 2.0),
                dim=1,
            )

        # Bernoulli MLP decoder: binary cross entropy loss
        if recon_2 is not None and x_2 is not None:
            loss_2 = torch.sum(self.logits_bce(recon_2, x_2), dim=1)

        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        if autoencoding:
            ELBO = torch.mean(lambda_1 * loss_1 + lambda_2 * loss_2 + beta * KLD)
        else:
            ELBO = torch.mean(beta * KLD)

        return ELBO

    def get_triplet_loss(self, mu_1, mu_2, mu_3, gt):
        """
        Compute the triplet margin loss.
        """
        positive_samples = gt == 1
        negative_samples = gt == 0
        anchors_size = min(torch.sum(positive_samples.int()), torch.sum(negative_samples.int()))
        positive_mask = positive_samples.repeat(1, mu_1.shape[1])
        negative_mask = negative_samples.repeat(1, mu_1.shape[1])

        triplet_loss = 0

        for mu in [mu_1, mu_2, mu_3]:
            positive_anchor = mu[positive_mask].view(torch.sum(positive_samples.int()), mu.shape[1])
            positive_anchor = positive_anchor[torch.randperm(positive_anchor.size()[0])]
            positive_anchor = positive_anchor[:anchors_size, :]

            negative_anchor = mu[negative_mask].view(torch.sum(negative_samples.int()), mu.shape[1])
            negative_anchor = negative_anchor[torch.randperm(negative_anchor.size()[0])]
            negative_anchor = negative_anchor[:anchors_size, :]

            positive_positive = positive_anchor[torch.randperm(positive_anchor.size()[0])]
            positive_negative = negative_anchor[torch.randperm(negative_anchor.size()[0])]
            negative_positive = negative_anchor[torch.randperm(negative_anchor.size()[0])]
            negative_negative = positive_anchor[torch.randperm(positive_anchor.size()[0])]

            triplet_loss += self.triplet_loss(positive_anchor, positive_positive, positive_negative) + \
                            self.triplet_loss(negative_anchor, negative_positive, negative_negative)

        return triplet_loss

    @staticmethod
    def save_checkpoint(state, folder='./', filename='model_best.pth.tar'):
        if not os.path.isdir(folder):
            os.mkdir(folder)
        print('Saving best model: epoch {}'.format(state['epoch']))
        torch.save(state, os.path.join(folder, filename))

    @staticmethod
    def load_checkpoint(checkpoint):
        model = MVIB(
            checkpoint['n_latents'],
            checkpoint['abundance_dim'],
            checkpoint['marker_dim'],
            checkpoint['device']
        )
        model.load_state_dict(checkpoint['state_dict'])

        return model, checkpoint['best_val_score']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
