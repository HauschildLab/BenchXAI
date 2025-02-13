"""
    Functions used during training and validation of AI models.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""
import os

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from bench_xai.misc.helper_functions import init_results_dic, load_metrics_to_device
from bench_xai.misc.results_save_functions import save_perf_dic_to_csv, save_nn_pred_probs_to_csv


def nn_data_loading(train_idx, val_idx, test_idx, data, batch_size):
    """
    Function for creating the train and data loader for pytorch neural networks.

    :param train_idx: indices of data used for training
    :param val_idx: indices of data used for validation
    :param test_idx: indices of data used for testing
    :param data: dataset
    :param batch_size: batch_size for training data
    :return: data loaders for training and validation data
    """

    # create samplers from indices
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sample = SubsetRandomSampler(test_idx)

    # Creating Pytorch data loaders
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)

    # validation loader should contain only one batch
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler)

    # test loader should contain only one batch
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_sample)

    return train_loader, validation_loader, test_loader


def train_nn(epochs, train_loader, optimizer, model, model_loss, dataset_name, model_type, n_classes=2,
             save_path=os.getcwd(), iteration=None, device='cuda'):
    """
    Training loop for Neural network training.

    :param epochs: number of epochs to train the model
    :param train_loader: data loader
    :param optimizer: optimizer used for backprop
    :param model: neural network model
    :param model_loss: loss function for training
    :param dataset_name: name of the dataset (used for saving)
    :param model_type: name of model_type (used for saving)
    :param device: device where calculations are running (default cuda)
    :param n_classes: number of target classes for classification (default=2 (binary))
    :param save_path: path where to save the results
    :param iteration: iteration of cross validation (default None)
    :return: saved metrics from training and trained model
    """
    model.to(device)

    # init results and metric dictionaries
    results = init_results_dic()
    metrics = load_metrics_to_device(n_classes, device)

    # training loop
    for epoch in range(epochs):  # loop over the dataset multiple times
        # loop over batches
        for i, (inputs, labels, filenames) in enumerate(train_loader, 0):
            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            outputs = model(inputs.to(device))
            # Find the Loss
            loss = model_loss(outputs, labels.to(device))
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # update metrics (if binary task only keep outputs for class 1)
            if outputs.shape[1]==2:
                outputs = outputs[:, 1]

            for key in metrics:
                metrics[key].update(outputs, labels.to(device))

        # calculate metric over all batches
        for key in metrics:
            results[key] = metrics[key].compute()

        save_perf_dic_to_csv(results, save_path, dataset_name + '_' + model_type + '_train_metrics_iteration' +
                             str(iteration) + '.csv', epoch, iteration=iteration)

        # reset metric
        for key in metrics:
            metrics[key].reset()

    return model


def validate_nn(validation_loader, model, dataset_name, model_type, n_classes=2, save_path=os.getcwd(), iteration=None,
                val_test = 'val', device='cuda'):
    """
    Validation loop for Neural networks.

    :param validation_loader: data loader
    :param model: neural network model
    :param dataset_name: name of the dataset (used for saving)
    :param model_type: name of model_type (used for saving)
    :param n_classes: number of target classes for classification (default=2 (binary))
    :param save_path: path where to save the results
    :param iteration: iteration of cross validation (default None)
    :param val_test: wether the filename should be called test or val (used for saving)
    :param device: device where calculations are running (default cuda)

    :return: saved metrics from validation, model outputs, model inputs, input labels, input file names (in case of
             non image data this is None)
    """
    model.to(device)
    # init results and metric dictionaries
    results = init_results_dic()
    metrics = load_metrics_to_device(n_classes, device)

    # validation loop over batches (should only be one batch with all validation data)
    for i, (inputs, labels, filenames) in enumerate(validation_loader, 0):
        # Forward Pass to get outputs
        outputs = model(inputs.to(device))

        # if binary task: only use outputs for class 1 for metric calculation
        if outputs.shape[1] == 2:
            outs = outputs[:, 1]
            # update metrics
            for key in metrics:
                metrics[key].update(outs, labels.to(device))
        else:
            # update metrics
            for key in metrics:
                metrics[key].update(outputs, labels.to(device))

        # check if last layer is softmax or logsoftmax
        if isinstance(list(model.children())[-1], torch.nn.Softmax) or isinstance(list(model.children())[-1], torch.nn.LogSoftmax):
            pred_probs = outputs
        # if not apply softmax to get pred probabilites
        else:
            pred_probs = outputs.softmax(dim=1)

        save_nn_pred_probs_to_csv(pred_probs, filenames, labels, save_path,
                                  dataset_name + '_' + model_type + '_' + val_test + '_pred_probabilities_iteration' + str(
                                      iteration) + '.csv', iteration=iteration)

    # compute validation metric (over all batches)
    for key in metrics:
        results[key] = metrics[key].compute()

    save_perf_dic_to_csv(results, save_path, dataset_name + '_' + model_type + '_' + val_test + '_metrics_iteration' +
                         str(iteration) + '.csv', epoch=None, iteration=iteration)
    # reset metric
    for key in metrics:
        metrics[key].reset()


def sklearn_data_loading(train_idx, val_idx, test_idx, data):
    """
    Function that returns training and testing data and labels for sklearn models.

    :param train_idx: indices of data used for training
    :param val_idx: indices of data used for validation
    :param test_idx: indices of data used for testing
    :param data: dataset

    :return: data loaders for training and validation data
    """

    x_train = data.data[train_idx, :]
    x_val = data.data[val_idx, :]
    x_test = data.data[test_idx, :]

    y_train = data.label[train_idx]
    y_val = data.label[val_idx]
    y_test = data.label[test_idx]

    return x_train, y_train, x_val, y_val, x_test, y_test


def train_sklearn(model, x_train, y_train, dataset_name, model_type, n_classes=2, save_path=os.getcwd(),
                  iteration=None, device='cuda'):
    """
    Training loop for training sklearn models.

    :param model: model from the sklearn library
    :param x_train: training data (without labels)
    :param y_train: labels for training data
    :param dataset_name: name of the dataset (used for saving)
    :param model_type: name of model_type (used for saving)
    :param n_classes: number of target classes for classification (default=2 (binary))
    :param save_path: path where to save the results
    :param iteration: iteration of cross validation (default None)
    :param device: device where calculations are running (default cuda)

    :return: saved metrics from training and trained model
    """
    # init results and metric dictionaries
    results = init_results_dic()
    metrics = load_metrics_to_device(n_classes, device)

    # train model
    model.fit(x_train, y_train)

    # test model on training data
    # if binary task only keep outputs for class 1
    if n_classes == 2:
        pred_proba = model.predict_proba(x_train)[:, 1]
    else:
        pred_proba = model.predict_proba(x_train)

    # calculate metric
    for key in metrics:
        results[key] = metrics[key](torch.tensor(pred_proba).to(device), y_train.to(device).to(torch.int8))

    save_perf_dic_to_csv(results, save_path, dataset_name + '_' + model_type + '_train_metrics_iteration' +
                         str(iteration) + '.csv', iteration=iteration)
    # reset metric
    for key in metrics:
        metrics[key].reset()

    return model


def validate_sklearn(model, x_test, y_test, dataset_name, model_type, n_classes=2, save_path=os.getcwd(),
                  iteration=None, val_test='val', device='cuda'):
    """
    Validation loop for sklearn models.

    :param model: model from the sklearn library
    :param x_test: validation data (without labels)
    :param y_test: labels for validation data
    :param dataset_name: name of the dataset (used for saving)
    :param model_type: name of model_type (used for saving)
    :param n_classes: number of target classes for classification (default=2 (binary))
    :param save_path: path where to save the results
    :param iteration: iteration of cross validation (default None)
    :param val_test: wether the filename should be called test or val (used for saving)
    :param device: device where calculations are running (default cuda)

    :return: saved metrics from training, prediction probabilities
    """
    # init results and metric dictionaries
    results = init_results_dic()
    metrics = load_metrics_to_device(n_classes, device)

    # test model on training data
    pred_proba = model.predict_proba(x_test)
    # if binary task only keep outputs for class 1
    if n_classes == 2:
        # calculate metric
        for key in metrics:
            results[key] = metrics[key](torch.tensor(pred_proba[:, 1]).to(device), y_test.to(device).to(torch.int8))
    else:
        # calculate metric
        for key in metrics:
            results[key]  = metrics[key](torch.tensor(pred_proba).to(device), y_test.to(device).to(torch.int8))

    save_perf_dic_to_csv(results, save_path, dataset_name + '_' + model_type + '_' + val_test + '_metrics_iteration' +
                         str(iteration) + '.csv', epoch=None, iteration=iteration)

    # reset metric
    for key in metrics:
        metrics[key].reset()

