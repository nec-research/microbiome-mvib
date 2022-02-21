"""
Trainer class.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import numpy as np
import timeit
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from microbiome_mvib.mvib import MVIB


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
        """
        :param model: Torch MVIB model object
        :param epochs: max training epochs
        :param lr: learning rate
        :param beta: multiplier for KL divergence / ELBO
        :param lambda_abundance: multiplier for abundance reconstruction loss
        :param lambda_markers: multiplier for markers reconstruction loss
        :param lambda_bce: multiplier for binary cross-entropy loss
        :param lambda_triplet: multiplier for triplet loss
        :param checkpoint_dir: directory for saving model checkpoints
        :param monitor: `min` minimize loss; `max` maximize ROC AUC
        """
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

        :param train_loader: the training Torch data loader
        :param val_loader: the validation Torch data loader
        :param bce: whether to optimize binary cross-entropy or not
        :param triplet: whether to optimize triplet loss or not
        :param autoencoding: whether to optimize reconstruction loss of not

        When autoencoding=True, the loss of the Multimodal Variational Autoencoder
        is optimized as well. See paper:
        https://papers.nips.cc/paper/2018/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf
        """
        if self.monitor == 'max':
            best_val_score = 0
        else:
            best_val_score = float('inf')
        is_best = False
        state = None

        t_0 = timeit.default_timer()

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

            if is_best or state is None:
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
                print('====> Epoch: {} | DeltaT: {} | Val score: {:.4f} | Best model'.format(
                    state['epoch'], timeit.default_timer() - t_0, best_val_score)
                )

        return state

    def train_step(self, train_loader, bce, triplet, autoencoding):
        """
        Compute loss and do backpropagation for a single epoch.
        """
        self.model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (abundance, markers, gt) in enumerate(train_loader):
            abundance_gt = Variable(abundance)
            markers_gt = Variable(markers)
            abundance = Variable(abundance)
            markers = Variable(markers)
            batch_size = len(abundance)

            # refresh the optimizer
            self.optimizer.zero_grad()

            # pass data through model
            recon_abundance_1, recon_markers_1, mu_1, logvar_1, cls_logits_1 = self.model(abundance, markers)
            recon_abundance_2, recon_markers_2, mu_2, logvar_2, cls_logits_2 = self.model(abundance)
            recon_abundance_3, recon_markers_3, mu_3, logvar_3, cls_logits_3 = self.model(markers=markers)

            # compute ELBO/KL divergence for each data combination
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

            # compute binary cross entropy for each data combination
            if bce:
                classification_loss = self.logits_bce(cls_logits_1, gt) + \
                                      self.logits_bce(cls_logits_2, gt) + \
                                      self.logits_bce(cls_logits_3, gt)
                classification_loss = torch.mean(classification_loss)
                train_loss = train_loss + self.lambda_bce * classification_loss

            # compute triplet loss for each data combination
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
        val_roc_auc = None
        gt_stack = np.array([])
        prediction_stack = np.array([])

        with torch.no_grad():
            for batch_idx, (abundance, markers, gt) in enumerate(val_loader):
                abundance_gt = Variable(abundance)
                markers_gt = Variable(markers)
                abundance = Variable(abundance)
                markers = Variable(markers)
                batch_size = len(abundance)

                recon_abundance_1, recon_markers_1, mu_1, logvar_1, cls_logits_1 = self.model(abundance, markers)
                recon_abundance_2, recon_markers_2, mu_2, logvar_2, cls_logits_2 = self.model(abundance)
                recon_abundance_3, recon_markers_3, mu_3, logvar_3, cls_logits_3 = self.model(markers=markers)

                # compute ELBO/KL divergence for each data combination
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

                # compute binary cross entropy for each data combination
                if bce:
                    classification_loss = self.logits_bce(cls_logits_1, gt) + \
                                          self.logits_bce(cls_logits_2, gt) + \
                                          self.logits_bce(cls_logits_3, gt)
                    classification_loss = torch.mean(classification_loss)
                    val_loss = val_loss + self.lambda_bce * classification_loss

                # compute triplet loss for each data combination
                if triplet:
                    triplet_loss = self.get_triplet_loss(mu_1, mu_2, mu_3, gt)
                    val_loss = val_loss + self.lambda_triplet * triplet_loss

                val_loss_meter.update(val_loss.item(), batch_size)

                gt_stack = np.concatenate(
                    (gt_stack, gt.cpu().numpy().squeeze().astype('int')),
                    axis=0)
                prediction_stack = np.concatenate(
                    (prediction_stack, torch.sigmoid(cls_logits_1).detach().cpu().numpy().squeeze()),
                    axis=0)

        if bce or triplet:
            val_roc_auc = roc_auc_score(gt_stack, np.nan_to_num(prediction_stack))

        return val_loss_meter.avg, val_roc_auc

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
