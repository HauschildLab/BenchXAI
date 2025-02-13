"""
    Function for training and validating model using Monte Carlo Cross Validation (MCCV).

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import os
import random

import torch
from bench_xai.misc.helper_functions import create_dir
from sklearn.model_selection import train_test_split
from bench_xai.misc import constants as const
from bench_xai.misc.results_save_functions import save_perf_dic_to_csv, save_nn_pred_probs_to_csv
from bench_xai.training.train_val_functions import train_nn, nn_data_loading, validate_nn, sklearn_data_loading, train_sklearn, \
    validate_sklearn

from bench_xai.xai_models.xai_loop import explain_and_eval, get_global_sklearn_relevances


def mccv_loop(result_path, data, dataset_name, data_type, model, model_type="pytorch", loss_func=None, epochs=None,
              batchsize=None, opti=None, mccv_start_iteration=0, mccv_end_iteration=100, validation_split=0.2, test_samples=50, random_seed=42,
              xai_methods=const.AVAILABLE_XAI_MODELS, train=True, config_path=None, device='cuda'):
    """
    Function where Monte Carlo Cross Validation (MCCV) is applied to data.
    For each MCCV iteration the data ist split randomly into training and validation data.
    For the validation data XAI relevances will be calculated for each MCCV iteration.

    :param result_path: path to results folder
    :param data: data loader
    :param dataset_name: name of the dataset
    :param data_type: type of data ('tabular_data','image_data','signal_data')
    :param model_type: type of ai model ('pytorch','logreg')
    :param model: model to train
    :param loss_func: loss function used for training (default=None)
    :param epochs: number of epochs to train NN for (None for other model types) (default=None)
    :param batchsize: batchsize of data used for training NN (None for other model types) (default=None)
    :param opti: optimizer used to train NN (None for other model types) (default=None)
    :param mccv_start_iteration: which iteration to start with (default=0)
    :param mccv_end_iteration: which iteration to end on (default=100)
                               used for iterating over range(mccv_start_iteration, mccv_end_iteration) (default [0,99])
    :param validation_split: split for training and validation (default=0.2, so 20% of data used for validation)
    :param random_seed: seed used for shuffling data during train/validation split (default=42)
    :param test_samples: integer or list of lists. amount of samples to take from each class for validating xai methods
                         (default=50) or list of sample indices. the list should contain two lists, the first with
                         names for samples of class 0 and the second with names for samples of class 1.
    :param xai_methods: xai methods that should be used for relevance calculation
    :param train: if the model should be trained or not (default: True)
    :param config_path: path to a config file (default: None), None uses default parameters for XAI methods
    :param device: where to put variables ('cuda' if available or 'cpu')
    """

    # assert correct model_type
    assert model_type in ['pytorch', 'rf', 'logreg'], f"Expected model_type='logreg', 'rf' or 'pytorch', got {model_type}."

    print('Training ', model_type, ' on ', dataset_name, '\n')

    # set path where to save results
    r_path = os.path.join(result_path, data_type, dataset_name)  # path where to save results

    assert isinstance(test_samples, list) or isinstance(test_samples, int), 'test_samples should be a list of ' \
                                                                            'filenames or indices (in case of tabular' \
                                                                            ' data) used for xai evaluation or an ' \
                                                                            'integer, e.g. 5 to choose 5 random ' \
                                                                            'samples of each classfor final ' \
                                                                            'testing.'

    if isinstance(test_samples, list):
        if data_type in ['image_data', 'signal_data']:
            # get test sample indices
            test_idx = [data.file_names.index(file) for file in test_samples]
            # get list with all 'indices'
            all_idx = list(range(data.__len__()))
            # remove test indices from all indices
            train_val_idx = [x for x in all_idx if x not in test_idx]
        elif data_type in ['tabular_data']:
            # get test sample indices
            test_idx = test_samples
            # get list with all 'indices'
            all_idx = list(range(data.__len__()))
            # remove test indices from all indices
            train_val_idx = [x for x in all_idx if x not in test_idx]

        # shuffle indices
        random.seed(random_seed)
        test_indices = random.sample(test_idx, len(test_idx))
        train_val_indices = random.sample(train_val_idx, len(train_val_idx))

    elif isinstance(test_samples, int):
        # init lists for test and train_val indices
        test_indices = []
        train_val_indices = []
        # iterate over all unique classes
        for cl in torch.unique(data.label):
            # get indices of class labels
            cl_idx = (data.label == int(cl)).nonzero().squeeze().tolist()

            # assert that enough samples of that class are available
            assert len(cl_idx) >= test_samples, f'Not enough samples of class {cl}.'

            # shuffle indices
            random.seed(random_seed)
            shuffled_cl_idx = random.sample(cl_idx, len(cl_idx))

            # split indices for testing and training+validation
            test_indices += shuffled_cl_idx[:test_samples]
            train_val_indices += shuffled_cl_idx[test_samples:]

    # [1. ] Perform Monte-Carlo-Cross-Validation -----------------------------------------------------------------------
    for it in range(mccv_start_iteration, mccv_end_iteration):
        print("Iteration ", it)

        # [3. ] Creating data indices for training and validation splits -----------------------------------------------
        # by shuffling and then performing a stratified split
        train_indices, val_indices = train_test_split(train_val_indices,
                                                      test_size=validation_split,
                                                      stratify=data.label[train_val_indices], shuffle=True,
                                                      random_state=random_seed * it)

        # training differs between pytorch and sklearn models
        if model_type == 'pytorch':
            # [4. ] Creating Dataloaders -------------------------------------------------------------------------------
            train_loader, validation_loader, test_loader = nn_data_loading(train_indices, val_indices, test_indices,
                                                                           data, batchsize)

            # [5. ] Init model, loss, optimizer, metrics ---------------------------------------------------------------
            net, loss, opti = model, loss_func, opti

            # [6. ] Train and validate Neural Network ------------------------------------------------------------------
            if train:
                trained_net = train_nn(epochs, train_loader, opti, net.to(device), loss, dataset_name, model_type, len(torch.unique(data.label)), r_path, it, device)
                create_dir(os.path.join(r_path, 'trained_models'))
                # save model
                torch.save(trained_net.state_dict(), os.path.join(r_path, 'trained_models', 'pytorch_model_iteration_' + str(it) + '.pth'))
                torch.save(trained_net.state_dict(), os.path.join(r_path, 'trained_models', 'pytorch_model_iteration_' + str(it) + '.pth'))
            else:
                trained_net = net.to(device)

            validate_nn(validation_loader, trained_net.to(device), dataset_name, model_type, len(torch.unique(data.label)), r_path, it, val_test='val', device=device)
            validate_nn(test_loader, trained_net.to(device), dataset_name, model_type, len(torch.unique(data.label)), r_path, it, val_test='test', device=device)

            # [8. ] get XAI relevances --------------------------------------------------------------------------------
            explain_and_eval(trained_net, test_loader, data_type, data, r_path, xai_methods, it, config_path, device)

# if the model is any sklearn model (e.g. log)
        elif model_type in ['logreg', 'rf']:

            # [9. ] Creating Datasets and init metrics ----------------------------------------------------------------
            x_train, y_train, x_val, y_val, x_test, y_test = sklearn_data_loading(train_indices, val_indices, test_indices, data)

            # [10. ] Train and validate LogReg model -------------------------------------------------------------------
            if train:
                t_results, trained_model = train_sklearn(model, x_train, y_train, dataset_name, model_type, len(torch.unique(data.label)), r_path, it, device)
            else:
                trained_model = model

            validate_sklearn(trained_model, x_val, y_val, dataset_name, model_type, len(torch.unique(data.label)), r_path, it, val_test='val', device=device)
            validate_sklearn(trained_model, x_test, y_test, dataset_name, model_type, len(torch.unique(data.label)), r_path, it, val_test='test', device=device)


            # [12. ] get XAI relevances --------------------------------------------------------------------------------
            get_global_sklearn_relevances(model, model_type, data.get_features(), r_path, it)
