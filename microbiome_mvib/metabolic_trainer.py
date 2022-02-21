"""
Trainer class - abundance, markers and metabolites.
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
            checkpoint_dir, monitor
    ):
        """
        :param model: Torch MVIB model object
        :param epochs: max training epochs
        :param lr: learning rate
        :param beta: multiplier for KL divergence / ELBO
        :param checkpoint_dir: directory for saving model checkpoints
        :param monitor: `min` minimize loss; `max` maximize ROC AUC
        """
        self.model = model
        self.epochs = epochs
        self.beta = beta
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.logits_bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.triplet_loss = torch.nn.TripletMarginLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, monitor, patience=10, cooldown=10, factor=0.5)

    def train(self, train_loader, val_loader):
        """
        The main training loop.

        :param train_loader: the training Torch data loader
        :param val_loader: the validation Torch data loader
        """
        if self.monitor == 'max':
            best_val_score = 0
        else:
            best_val_score = float('inf')
        is_best = False
        state = None

        t_0 = timeit.default_timer()

        for epoch in range(1, self.epochs + 1):
            self.train_step(train_loader)
            val_loss, val_roc_auc = self.evaluate(val_loader)

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
                    'metabolite_dim': self.model.metabolite_dim,
                    'device': self.model.device,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch
                }
                print('====> Epoch: {} | DeltaT: {} | Val score: {:.4f} | Best model'.format(
                    state['epoch'], timeit.default_timer() - t_0, best_val_score)
                )

        return state

    def train_step(self, train_loader):
        """
        Compute loss and do backpropagation for a single epoch.
        """
        self.model.train()
        train_loss_meter = AverageMeter()
        train_loss = 0

        for batch_idx, (abundance, markers, metabolite, gt) in enumerate(train_loader):
            abundance = Variable(abundance)
            markers = Variable(markers)
            batch_size = len(abundance)

            # refresh the optimizer
            self.optimizer.zero_grad()

            # pass data through model
            _, __, mu_1, logvar_1, cls_logits_1 = self.model(abundance, markers, metabolite)
            _, __, mu_2, logvar_2, cls_logits_2 = self.model(abundance)
            _, __, mu_3, logvar_3, cls_logits_3 = self.model(markers=markers)
            _, __, mu_4, logvar_4, cls_logits_4 = self.model(metabolite=metabolite)
            _, __, mu_5, logvar_5, cls_logits_5 = self.model(abundance, markers)
            _, __, mu_6, logvar_6, cls_logits_6 = self.model(abundance, metabolite=metabolite)
            _, __, mu_7, logvar_7, cls_logits_7 = self.model(markers=markers, metabolite=metabolite)

            mus = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
            logvars = [logvar_1, logvar_2, logvar_3, logvar_4, logvar_5, logvar_6, logvar_7]
            class_logits = [
                cls_logits_1,
                cls_logits_2,
                cls_logits_3,
                cls_logits_4,
                cls_logits_5,
                cls_logits_6,
                cls_logits_7
            ]

            # compute KL divergence for each data combination
            for i in range(len(mus)):
                train_loss += self.KL_loss(mus[i], logvars[i], self.beta)

            # compute binary cross entropy for each data combination
            classification_loss = 0
            for i in range(len(mus)):
                classification_loss += self.logits_bce(class_logits[i], gt)
            classification_loss = torch.mean(classification_loss)
            train_loss += classification_loss

            train_loss_meter.update(train_loss.item(), batch_size)

            # compute gradients and take step
            train_loss.backward()
            self.optimizer.step()

    def evaluate(self, val_loader):
        """
        Evaluate model performance on validation (or test) set.
        """
        self.model.eval()
        val_loss_meter = AverageMeter()
        val_roc_auc = None
        gt_stack = np.array([])
        prediction_stack = np.array([])
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (abundance, markers, metabolite, gt) in enumerate(val_loader):

                abundance = Variable(abundance)
                markers = Variable(markers)
                batch_size = len(abundance)

                # pass data through model
                _, __, mu_1, logvar_1, cls_logits_1 = self.model(abundance, markers, metabolite)
                _, __, mu_2, logvar_2, cls_logits_2 = self.model(abundance)
                _, __, mu_3, logvar_3, cls_logits_3 = self.model(markers=markers)
                _, __, mu_4, logvar_4, cls_logits_4 = self.model(metabolite=metabolite)
                _, __, mu_5, logvar_5, cls_logits_5 = self.model(abundance, markers)
                _, __, mu_6, logvar_6, cls_logits_6 = self.model(abundance, metabolite=metabolite)
                _, __, mu_7, logvar_7, cls_logits_7 = self.model(markers=markers, metabolite=metabolite)

                mus = [mu_1, mu_2, mu_3, mu_4, mu_5, mu_6, mu_7]
                logvars = [logvar_1, logvar_2, logvar_3, logvar_4, logvar_5, logvar_6, logvar_7]
                class_logits = [
                    cls_logits_1,
                    cls_logits_2,
                    cls_logits_3,
                    cls_logits_4,
                    cls_logits_5,
                    cls_logits_6,
                    cls_logits_7
                ]

                # compute KL divergence for each data combination
                for i in range(len(mus)):
                    val_loss += self.KL_loss(mus[i], logvars[i], self.beta)

                # compute binary cross entropy for each data combination
                classification_loss = 0
                for i in range(len(mus)):
                    classification_loss += self.logits_bce(class_logits[i], gt)
                classification_loss = torch.mean(classification_loss)
                val_loss += classification_loss

                val_loss_meter.update(val_loss.item(), batch_size)

                gt_stack = np.concatenate(
                    (gt_stack, gt.cpu().numpy().squeeze().astype('int')),
                    axis=0)
                prediction_stack = np.concatenate(
                    (prediction_stack, torch.sigmoid(cls_logits_1).detach().cpu().numpy().squeeze()),
                    axis=0)

        val_roc_auc = roc_auc_score(gt_stack, np.nan_to_num(prediction_stack))

        return val_loss_meter.avg, val_roc_auc

    def KL_loss(self, mu, logvar, beta=1):
        """
        This method returns the KL divergence.
        """
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return torch.mean(beta * KLD)

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
            checkpoint['device'],
            checkpoint['metabolite_dim']
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
