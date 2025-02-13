"""
    Small helper functions to reduce code length.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import os
import copy
import torchmetrics

import numpy as np

from bench_xai.misc.constants import PERFORMANCE_METRICS
import torch


def create_dir(path):
    """
    Check if path exists. If not create it.
    :param path: folder path that will be created
    :return also return the path after saving
    """

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def load_metrics_to_device(num_classes, device):
    """
    Function that loads metrics for measuring model performance to specified device.
    In case of 1 neuron output torchmetrics automatically applies sigmoid if network output is logits.
    In case of >1 neuron output torchmetrics applies argmax to get class with highest logit/probability.

    So there is no need to apply softmax or sigmoid to logit outputs before giving them to torchmetrics.

    :param device: either gpu or cpu
    :param num_classes: number of classes to classify
    :return: metrics on specified device
    """
    metrics = copy.deepcopy(PERFORMANCE_METRICS)

    # by default, it's binary classification
    task = "binary"
    # if more than 2 classes are present then switch to multiclass classification
    # in case of multiclass metric is calculated for each label and then averaged to one final value
    # see torchmetrics documentation for more information
    if num_classes > 2:
        task = "multiclass"

    met_dic = {}
    for met in metrics:
        if met == 'Accuracy':
            met_dic[met] = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(device)
        elif met == 'AUROC':
            met_dic[met] = torchmetrics.AUROC(task=task, num_classes=num_classes).to(device)
        elif met == 'Sensitivity':
            met_dic[met] = torchmetrics.Recall(task=task, num_classes=num_classes).to(device)
        elif met == 'Specificity':
            met_dic[met] = torchmetrics.Specificity(task=task, num_classes=num_classes).to(device)
        elif met == 'F1':
            met_dic[met] = torchmetrics.F1Score(task=task, num_classes=num_classes).to(device)
        elif met == 'MCC':
            met_dic[met] = torchmetrics.MatthewsCorrCoef(task=task, num_classes=num_classes).to(device)
        elif met == 'AUPRC':
            met_dic[met] = torchmetrics.AveragePrecision(task=task, num_classes=num_classes).to(device)

    return met_dic


def init_results_dic():
    """
    Function that returns dictionary to save all training and validation metric results.
    """
    results = {}
    for met in PERFORMANCE_METRICS:
        results[met] = []

    return results


def set_rule_attr(layer, attr):
    """
    Function that recursively sets rule attribute for LRP methods.

    :param layer: Layer of pytorch model
    :param attr: Attribute for LRP model to be set as a rule
    """

    if len(list(layer.children())) == 0:
        setattr(layer, "rule", attr)
    else:
        for sublayer in layer.children():
            set_rule_attr(sublayer, attr)


def it_batch(iterable, n=1):
    """
    Function that takes an iterable and batches it into smaller iterables.

    :param iterable: Iterable variable
    :param n: batchsize
    """

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def create_super_pixel_mask(input_shape, superpixel_dim, device):
    """
    Function that creates a super pixel mask the same shape as inputs.

    :param input_shape: shape of the inputs. either (batch_dim, col, height, width) or (batch_dim, height, width). First
                        dimension should always be batch_dim.
    :param superpixel_dim: dimension of the super pixel (should only be (height, width)) never 3
    :param device: device where to move output (default: cuda)

    :return: super pixel mask
    """

    # check if color channel dimension is present (input_shape should have length 4)
    if len(input_shape) == 4:
        # get number of features for the width
        nf_width = int(np.ceil(input_shape[3] / superpixel_dim[1]))
        # get number of features for the height
        nf_height = int(np.ceil(input_shape[2] / superpixel_dim[0]))
        # get total number of features
        n_features = nf_width * nf_height
        # get list of super pixels
        super_pixels = [np.reshape(np.repeat(i, superpixel_dim[1] * superpixel_dim[0]), superpixel_dim) for i in
                        range(n_features)]
        # turn it into mask of superpixels by first hstacking superpixels and then vstacking the hstacks
        mask = np.vstack(
            [np.hstack(super_pixels[0 + nf_width * l:nf_width + nf_width * l]) for l in range(nf_height)])

        # calculate the amount to be removed from the last row and column of superpixels
        rem_height = mask.shape[0] - input_shape[2]
        rem_width = mask.shape[1] - input_shape[3]
        # remove them if needed
        if rem_height != 0:
            if rem_width != 0:
                mask = mask[:-rem_height, :-rem_width]
            else:
                mask = mask[:-rem_height, :]
        else:
            if rem_width != 0:
                mask = mask[:, :-rem_width]
            else:
                mask = mask[:, :]

        # create color channels
        mask3d = np.repeat([mask], input_shape[1], axis=0)
        # create batch dim
        super_pixel_mask = torch.from_numpy(np.repeat([mask3d], input_shape[0], axis=0)).to(torch.long).to(device)

    else:
        # get number of features for the width
        nf_width = int(np.ceil(input_shape[2] / superpixel_dim[1]))
        # get number of features for the height
        nf_height = int(np.ceil(input_shape[1] / superpixel_dim[0]))
        # get total number of features
        n_features = nf_width * nf_height

        # get list of super pixels
        super_pixels = [np.reshape(np.repeat(i, superpixel_dim[0] * superpixel_dim[1]), superpixel_dim) for i in
                        range(n_features)]
        # turn it into mask of superpixels by first hstacking superpixels and then vstacking the hstacks
        mask = np.vstack([np.hstack(super_pixels[0 + nf_width * l:nf_width + nf_width * l]) for l in range(nf_height)])

        # calculate the amount to be removed from the last row and column of superpixels
        rem_height = mask.shape[0] - input_shape[1]
        rem_width = mask.shape[1] - input_shape[2]
        # remove them if needed
        if rem_height != 0:
            if rem_width != 0:
                mask = mask[:-rem_height, :-rem_width]
            else:
                mask = mask[:-rem_height, :]
        else:
            if rem_width != 0:
                mask = mask[:, :-rem_width]
            else:
                mask = mask[:, :]

        # create batch dim
        super_pixel_mask = torch.from_numpy(np.repeat([mask], input_shape[0], axis=0)).to(torch.long).to(device)

    return super_pixel_mask