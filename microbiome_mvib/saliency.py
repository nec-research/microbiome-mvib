"""
The Saliency class.
"""
import os
import torch
import numpy as np


class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        tensor = tensor.double()  # in double for numerical stability
        scale = 1.0 / (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        tensor = tensor.float()
        return tensor


class Saliency:
    """
    Implements "Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps"
    Karen Simonyan, Andrea Vedaldi, Andrew Zisserman
    2014

    Paper: https://arxiv.org/abs/1312.6034
    """
    def __init__(self, dataset):
        self._dataset = dataset

        # for positive predictions
        self._abundance_gradients = None
        self._markers_gradients = None

        # for the full dataset
        self._full_markers_gradients = None
        self._full_abundance_gradients = None
        self._labels = None
        self._predictions = None

    def init(self):
        """
        Initialization: input needs gradients for saliency.
        """
        self._dataset.abundance.requires_grad_()
        self._dataset.markers.requires_grad_()

    def stop(self):
        """
        Do not store gradients for inputs in general.
        """
        self._dataset.abundance.requires_grad = False
        self._dataset.markers.requires_grad = False
        self._dataset.abundance.grad = None
        self._dataset.markers.grad = None

    def update(self, prediction, labels):
        """
        Compute saliency for abundances and markers profiles by backpropagating the sick patient predictions.
        """
        sick_predictions = prediction[prediction > 0.5]
        sick_predictions.mean().backward()

        # get absolute values of the gradients
        abundance_grad = self._dataset.abundance.grad.cpu().abs().data
        markers_grad = self._dataset.markers.grad.cpu().abs().data

        if self._full_abundance_gradients is None or self._full_markers_gradients is None:
            self._full_abundance_gradients = abundance_grad
            self._full_markers_gradients = markers_grad
            self._labels = labels.detach().cpu()
            self._predictions = prediction.detach().cpu()
        else:
            self._full_abundance_gradients = torch.cat((
                self._full_abundance_gradients,
                abundance_grad
            ))
            self._full_markers_gradients = torch.cat((
                self._full_markers_gradients,
                markers_grad
            ))
            self._labels = torch.cat((
                self._labels,
                labels.detach().cpu()
            ))
            self._predictions = torch.cat((
                self._predictions,
                prediction.detach().cpu()
            ))

    def save(self, dir):
        """
        Save the gradients computed by Saliency for both markers and species.
        Additionally, save predictions and labels, so that one can extract
        saliency maps for e.g. true positive predictions.
        """
        np.save(
            os.path.join(dir, 'full_abundance_saliency.npy'),
            self._full_abundance_gradients
        )
        np.save(
            os.path.join(dir, 'full_markers_saliency.npy'),
            self._full_markers_gradients
        )
        np.save(
            os.path.join(dir, 'full_predictions.npy'),
            self._predictions
        )
        np.save(
            os.path.join(dir, 'full_labels.npy'),
            self._labels
        )
