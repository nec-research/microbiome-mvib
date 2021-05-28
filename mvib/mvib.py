# coding=utf-8
#        Multimodal Variational Information Bottleneck
#
#   File:     mvib.py
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
Multimodal Variational Information Bottleneck.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MVIB(nn.Module):
    """
    Multimodal Variational Information Bottleneck.
    """
    def __init__(self, n_latents, abundance_dim, marker_dim, device):
        super(MVIB, self).__init__()

        self.abundance_encoder = AbundanceEncoder(x_dim=abundance_dim, z_dim=n_latents)
        self.markers_encoder = MarkersEncoder(x_dim=marker_dim, z_dim=n_latents)

        self.abundance_decoder = GaussianDecoder(x_dim=abundance_dim, z_dim=n_latents)
        self.markers_decoder = BernoulliDecoder(x_dim=marker_dim, z_dim=n_latents)

        self.classifier = nn.Linear(n_latents, 1)
            
        self.experts = ProductOfExperts()

        self.n_latents = n_latents
        self.abundance_dim = abundance_dim
        self.marker_dim = marker_dim
        self.device = device

    def reparametrize(self, mu, logvar):
        """
        Reparameterization trick.
        Samples z from its posterior distribution.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, abundance=None, markers=None):
        # infer joint posterior
        mu, logvar = self.infer(abundance, markers)

        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # autoencoding
        abundance_recon = self.abundance_decoder(z)
        markers_recon = self.markers_decoder(z)

        # classification
        classification_logits = self.classifier(z)

        return abundance_recon, markers_recon, mu, logvar, classification_logits

    def infer(self, abundance=None, markers=None):
        """
        Infer joint posterior q(z|x).
        """
        batch_size = abundance.size(0) if abundance is not None else markers.size(0)
        # initialize the universal prior expert
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents))
        if abundance is not None:
            a_mu, a_logvar = self.abundance_encoder(abundance)
            mu = torch.cat((mu, a_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, a_logvar.unsqueeze(0)), dim=0)

        if markers is not None:
            m_mu, m_logvar = self.markers_encoder(markers)
            mu = torch.cat((mu, m_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, m_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

    def classify(self, abundance=None, markers=None):
        """
        Classification - Compute p(y|x).
        """
        mu, logvar = self.infer(abundance, markers)
        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)
        classification_logits = self.classifier(z)
        prediction = torch.sigmoid(classification_logits)
        return prediction

    def prior_expert(self, size):
        """
        Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        :param size: dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        mu, logvar = mu.to(self.device), logvar.to(self.device)
        return mu, logvar


class AbundanceEncoder(nn.Module):
    """
    Parametrizes q(z|x).
    :param x_dim: input dimension
    :param z_dim: latent dimension
    """
    def __init__(self, x_dim, z_dim):
        super(AbundanceEncoder, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, x_dim // 2)
        self.fc2 = nn.Linear(x_dim // 2, x_dim // 2)
        self.fc31 = nn.Linear(x_dim // 2, z_dim)
        self.fc32 = nn.Linear(x_dim // 2, z_dim)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        h = self.drop(F.silu(self.fc1(x.view(-1, self.x_dim))))
        h = self.drop(F.silu(self.fc2(h)))
        return self.fc31(h), self.fc32(h)


class MarkersEncoder(nn.Module):
    """
    Parametrizes q(z|x).
    :param x_dim: input dimension
    :param z_dim: latent dimension
    """
    def __init__(self, x_dim, z_dim):
        super(MarkersEncoder, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc31 = nn.Linear(1024, z_dim)
        self.fc32 = nn.Linear(1024, z_dim)
        self.drop = nn.Dropout(p=0.4)

    def forward(self, x):
        h = self.drop(F.silu(self.fc1(x)))
        h = self.drop(F.silu(self.fc2(h)))
        return self.fc31(h), self.fc32(h)


class BernoulliDecoder(nn.Module):
    """
    Parametrizes p(x|z) for a Bernoulli distribution.
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    :param z_dim: number of latent dimensions
    :param x_dim: output dimension
    """
    def __init__(self, z_dim, x_dim):
        super(BernoulliDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, x_dim)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, z):
        h = self.bn1(F.relu(self.fc1(z)))
        return self.fc2(h)


class GaussianDecoder(nn.Module):
    """
    Parametrizes p(x|z) for a Gaussian distribution.
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    :param z_dim: number of latent dimensions
    :param x_dim: output dimension
    """
    def __init__(self, z_dim, x_dim):
        super(GaussianDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, z_dim * 2)
        self.fc2 = nn.Linear(z_dim * 2, x_dim)
        self.fc3 = nn.Linear(z_dim * 2, x_dim)

    def forward(self, z):
        h = torch.tanh(self.fc1(z))
        mu = self.fc2(h)
        sigma = self.fc3(h)
        return mu, sigma


class ProductOfExperts(nn.Module):
    """
    Compute parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    """
    def forward(self, mu, logvar, eps=1e-8):
        """
        :param mu: M x D for M experts
        :param logvar: M x D for M experts
        :param eps: constant for stability
        """
        # do computation in double to avoid nans
        mu = mu.double()
        logvar = logvar.double()

        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)

        # back to float
        pd_mu = pd_mu.float()
        pd_logvar = pd_logvar.float()

        return pd_mu, pd_logvar


class Ensembler:
    """
    A class to ensemble model predictions and latent representations from multiple models.
    """
    def __init__(self):
        self._mu = 0
        self._prediction = 0
        self._count = 0

    def add(self, mu, prediction):
        """
        This method needs to be called to add a model to the ensemble.

        :param mu: Torch tensor of the z Gaussian mean
        :param prediction: Torch tensor with the model predictions on the classification task
        """
        self._mu += mu.cpu().detach().numpy()
        self._prediction += prediction.cpu().detach().numpy()
        self._count += 1

    def avg(self):
        """
        This method averages the models in the ensemble.
        """
        self._mu = self._mu / self._count
        self._prediction = self._prediction / self._count

        return self._mu, self._prediction
