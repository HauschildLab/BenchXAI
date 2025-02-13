"""
    Functions for calculating XAI relevances and global logreg coefficients.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import os
import torch

# import helper functions
from bench_xai.misc.helper_functions import create_dir, it_batch
from bench_xai.misc.results_save_functions import save_rel_dic, save_global_coef_dic, save_image_rel, save_time_seconds

# import xai models
from bench_xai.xai_models.xai_models import get_rel
import bench_xai.misc.constants as const


def explain_and_eval(model, data_loader, data_type, data, results_path,
                     xai_methods=const.AVAILABLE_XAI_MODELS, iteration=None, config_path=None, device='cuda'):
    """
    Function that first divides samples used for model validation into classified as TP, TN, FP, FN and then calculates
    relevances with given XAI methods for each sample and saves them in .csv files.

    :param model: AI model used for training and validation
    :param data_loader: dataloader to load samples
    :param data_type: data type (either 'image_data', 'tabular_data', 'signal_data')
    :param data: dataset
    :param results_path: path to the result folder
    :param xai_methods: list of names of the XAI models to be used.
           Default: All available XAI methods (see misc.constants.py)
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. Default = None
    :param config_path: path to a config file (default: None), None uses default parameters for XAI methods
    :param device: where to put variables (default 'cuda')
    """

    # [2. ] get output labels and target labels ------------------------------------------------------------------------

    #labs = labels.to(device)
    #list_filenames = [filenames[i] for i in range(len(filenames))]

    for xai in xai_methods:
        # [6. ] create folders to save results to ------------------------------------------------------------------
        xai_path = os.path.join(results_path, 'relevances', xai)

        # validation loop over batches (should only be one batch with all validation data)
        for i, (inputs, labels, filenames) in enumerate(data_loader, 0):
            # [8. ] Get dictionary of XAI attributions and save them -------------------------------------------
            # init dict for relevance attributions
            rels = {}
            # appen filenames to rel_dic
            rels['IDs'] = filenames

            # get relevances from selected xai_model
            batch_attr, batch_time_sec = get_rel(xai, model, data_type, inputs.to(device), labels.to(device), config_path, device)

            if data_type == 'tabular_data':
                batch_attr = batch_attr.cpu().detach().numpy()
                feature_names = data.get_features()
                for i in range(len(feature_names)):
                    # append feat_rel to feat_names
                    rels[feature_names[i]] = []
                    # get relevance values for each feature
                    feat_rel = [r[i] for r in batch_attr]
                    # append feat_rel to feat_names
                    rels[feature_names[i]] = feat_rel

                save_rel_dic(rels, xai, xai_path, iteration)
                save_time_seconds(results_path, xai, batch_time_sec, len(inputs), inputs.shape[1], iteration)

            elif data_type in ['image_data', 'signal_data']:
                save_image_rel(batch_attr, filenames, xai, results_path, iteration)

                # check if image is 3D (batch_size, color, x_dim, y_dim) or 2D (batch_size, x_dim, y_dim)
                if len(inputs.shape) == 4:
                    save_time_seconds(results_path, xai, batch_time_sec, len(inputs), (inputs.shape[2], inputs.shape[3]),
                                      iteration)
                else:
                    save_time_seconds(results_path, xai, batch_time_sec, len(inputs), (inputs.shape[1], inputs.shape[2]),
                                      iteration)


def get_global_sklearn_relevances(model, model_type, feature_names, results_path, iteration=None):
    """
    Function that returns global model coefficients.

    :param model: AI model used for training and validation
    :param model_type: type of model
    :param feature_names: names of features
    :param results_path: path to the result folder
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)

    :return:
    """
    # init dict for relevances
    rel = {}
    # path to save sklearn model relevances to
    xai_path = create_dir(os.path.join(results_path, 'relevances', model_type))

    if model_type == 'logreg':
        rels = model.coef_
        rels = rels[0]  # np.ndarray of shape (n_features,)
    elif model_type == 'rf':
        rels = model.feature_importances_  # np.ndarray of shape (n_features,)

    # iterate over features
    for i in range(len(feature_names)):
        # append relevance to feature as list
        rel[feature_names[i]] = [rels[i]]

    save_global_coef_dic(rel, model_type, xai_path, iteration)
