"""
    Functions that save results in csv formats
    in correct folders.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import csv
import os

import pandas as pd
import torch

from bench_xai import create_dir


def save_perf_dic_to_csv(dic, path, filename, epoch=None, iteration=None):
    """
    Function that saves performance metrics values from dictionary in csv files.

    :param dic: dictionary containing performance metrics results
    :param path: path to file where results will be saved
    :param filename: name of the file which to save the results into
    :param epoch: Integer or None, if results are calculated for every epoch of training (only for NN)
                   (default = True)
    :param iteration: None if the model is only trained once, Integer if the model is trained multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """
    # create metrics folder in results folder if it doesn't exist already
    create_dir(os.path.join(path, 'metrics'))
    f_path = os.path.join(path, 'metrics', filename)

    # turn the metrics dictionary into a list of sublists
    values = [value for value in dic.values()]

    # check if the csv file should be created for multiple iterations or just one
    if iteration is not None:
        # only write the header the first iteration
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            if epoch is not None:
                header = ['Iteration', 'Epoch'] + list(dic.keys())
            else:
                header = ['Iteration'] + list(dic.keys())

            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append all results for every epoch
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # only NN have epochs
            if epoch is not None:
                # append iteration, epoch and metrics to one list
                data_per_epoch = [iteration] + [epoch] + [val.item() for val in values]
                # write list to csv
                w.writerow(data_per_epoch)
            else:
                # append iteration, epoch and metrics to one list
                data = [iteration] + [val.item() for val in values]
                # write list to csv
                w.writerow(data)
    # in case the model is only trained once just save the performance results
    else:
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            if epoch is not None:
                header = ['Epoch'] + list(dic.keys())
            else:
                header = list(dic.keys())

            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append all results for every epoch
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # only NN have epochs
            if epoch is not None:
                # append iteration, epoch and metrics to one list
                data_per_epoch = [epoch] + [val.item() for val in values]
                # write list to csv
                w.writerow(data_per_epoch)
            else:
                # append iteration, epoch and metrics to one list
                data = [val.item() for val in values]
                # write list to csv
                w.writerow(data)


def save_nn_pred_probs_to_csv(pred_probs, samp_ids, labels, path, filename, iteration=None):
    """
    Function that saves performance metrics values from dictionary in csv files.

    :param pred_probs: tensor with pred probabilities for all samples
    :param samp_ids: tuple with sample ids for all samples
    :param labels: tensor with labels for all samples
    :param path: path to file where results will be saved
    :param filename: name of the file which to save the results into
    :param iteration: None if the model is only trained once, Integer if the model is trained multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """

    # create dictionary with ['sample_id', 'true_label', 'Probs_Class_0', ...., 'Probs_Class_n']
    probs_dic = {}
    # check if samp ids comes from file names or not
    if '.' in list(samp_ids)[0]:
        probs_dic['IDs'] = [''.join(list(samp_ids)[img].split('.')[:-1]) for img in range(len(list(samp_ids)))]
    else:
        probs_dic['IDs'] = list(samp_ids)
    probs_dic['True_Labels'] = labels.tolist()

    # iterate over all classes
    for cl in range(pred_probs.shape[1]):
        # create dict key for each class
        key = 'Probabilities_Class_' + str(cl)
        # get class probabilites for each class
        probs_dic[key] = pred_probs[:, cl].tolist()

    # create metrics folder in results folder if it doesn't exist already
    create_dir(os.path.join(path, 'pred_probs'))
    f_path = os.path.join(path, 'pred_probs', filename)

    # turn the metrics dictionary into a list of sublists
    values = [value for value in probs_dic.values()]

    # check if the csv file should be created for multiple iterations or just one
    if iteration is not None:
        # only write the header the first iteration
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            header = ['Iteration'] + list(probs_dic.keys())

            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append all results for every epoch
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # save pred probs for each sample
            for sample in range(len(values[0])):
                data = [iteration] + [val[sample] for val in values]
                # write list to csv
                w.writerow(data)
    # in case the model is only trained once just save the performance results
    else:
        if not os.path.exists(f_path):
            header = list(probs_dic.keys())
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append all results for every epoch
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # save pred probs for each sample
            for sample in range(len(values[0])):
                data = [val[sample] for val in values]
                # write list to csv
                w.writerow(data)


def save_rel_dic(rel_dic, xai_model, xai_path, iteration=None):
    """
    Function that saves relevance values from dictionary in csv files.

    :param rel_dic: dictionary with relevances for each feature
    :param xai_model: name of the XAI model used to generate relevances
    :param xai_path: path where to save relevances
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """
    create_dir(xai_path)

    # path to save file (e.g. '../Integrated_Gradients/Integrated_Gradients_relevances.csv')
    f_path = os.path.join(xai_path, xai_model + '_relevances_iteration' + str(iteration) + '.csv')

    # turn the metrics dictionary into a list of sublists
    values = [value for value in rel_dic.values()]


    # check if the csv file should be created for multiple iterations or just one
    if iteration is not None:
        # only write the header the first iteration
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            header = ['Iteration'] + list(rel_dic.keys())
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append relevances for current iteration
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # iterate over samples
            for sample in range(len(values[0])):
                # append iteration, epoch and metrics to one list
                rel_per_sample = [iteration] + [val[sample] for val in values]
                # write list to csv
                w.writerow(rel_per_sample)
    else:
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            header = list(rel_dic.keys())
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append relevances for current iteration
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # iterate over samples
            for sample in range(len(values[0])):
                # append iteration, epoch and metrics to one list
                rel_per_sample = [val[sample] for val in values]
                # write list to csv
                w.writerow(rel_per_sample)


def save_global_coef_dic(rel_dic, xai_model, xai_path, iteration=None):
    """
    Function that saves relevances of log_reg from dictionary in csv files.

    :param rel_dic: dictionary with relevances for each feature
    :param xai_model: name of the XAI model used to generate relevances
    :param xai_path: path where to save relevances
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """

    # path to save file
    f_path = os.path.join(xai_path, xai_model + '_global_relevances_iteration' + str(iteration) + '.csv')

    # turn the metrics dictionary into a list of sublists
    values = [value for value in rel_dic.values()]


    # check if the csv file should be created for multiple iterations or just one
    if iteration is not None:
        # only write the header the first iteration
        if iteration == 0:
            # create the header for the csv file (only NN have epochs column)
            header = ['Iteration'] + list(rel_dic.keys())
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append relevances for current iteration
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # iterate over samples
            for sample in range(len(values[0])):
                # append iteration, epoch and metrics to one list
                rel_per_sample = [iteration] + [val[sample] for val in values]
                # write list to csv
                w.writerow(rel_per_sample)
    else:
        # create the header for the csv file (only NN have epochs column)
        header = list(rel_dic.keys())
        # create the csv file with the header
        with open(f_path, 'w', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # write the header
            w.writerow(header)

        # append relevances for current iteration
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # iterate over samples
            for sample in range(len(values[0])):
                # append iteration, epoch and metrics to one list
                rel_per_sample = [val[sample] for val in values]
                # write list to csv
                w.writerow(rel_per_sample)


def save_image_rel(relevances, list_filenames, xai_model, r_path, iteration=None):
    """
    Function that saves original image as well as relevances into a csv file.
    This way the relevances can later be mapped to the correct image.

    :param relevances: XAI relevances as pytorch tensor
    :param xai_model: name of the XAI model used to generate relevances
    :param list_filenames: list containing filenames for inputs to be saved
    :param r_path: path to the result folder
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """

    # create "relevances" folder to save relevance attributions into
    results_path = os.path.join(r_path, 'relevances')
    create_dir(results_path)

    for img in range(len(list_filenames)):
        # remove datatype ending from file name (e.g. cat0.jpg -> cat0)
        img_without_file_ending = ''.join(list_filenames[img].split('.')[:-1])

        # path to image specific results
        img_path = os.path.join(results_path, img_without_file_ending)
        create_dir(img_path)
        # path to xai method in image folder
        xai_path = os.path.join(results_path, img_without_file_ending, xai_model)
        create_dir(xai_path)

        # check if the pkl file should be created for multiple iterations or just one
        if iteration is not None:
            # path to pkl file here relevances for image will be save
            # (e.g. '..cat0/Integrated_Gradients/Integrated_Gradients_FN_relevances_Iteration_0.csv')
            f_path = os.path.join(xai_path, xai_model + '_relevances_Iteration_' + str(iteration) + '.csv')

            # check if image is 3D (color, x_dim, y_dim) or 2D (x_dim, y_dim)
            if len(relevances[img].shape) == 3:
                # .clone().detach().permute(1, 2, 0) permutes the color channels to the back (for visualisation later)
                data = relevances[img].clone().detach().cpu().permute(1, 2, 0)
                data_2d = torch.sum(data, dim=2)  # sum over color channel so that we have a 2d heatmap in the end
                data_df = pd.DataFrame(data_2d.numpy())
            else:
                data_df = pd.DataFrame(relevances[img].clone().detach().cpu())

            data_df.to_csv(f_path, index=False)  # save to file

        else:
            # path to save file (e.g. '..cat0/Integrated_Gradients/Integrated_Gradients_FN_relevances.csv')
            f_path = os.path.join(xai_path, xai_model + '_relevances.csv')
            # check if image is 3D (color, x_dim, y_dim) or 2D (x_dim, y_dim)
            if len(relevances[img].shape) == 3:
                # .clone().detach().permute(1, 2, 0) permutes the color channels to the back (for visualisation later)
                data = relevances[img].clone().detach().cpu().permute(1, 2, 0)
                data_2d = torch.sum(data, dim=2)  # sum over color channel so that we have a 2d heatmap in the end
                data_df = pd.DataFrame(data_2d.numpy())
            else:
                data_df = pd.DataFrame(relevances[img].clone().detach().cpu())

            data_df.to_csv(f_path, index=False)  # save to file


def save_time_seconds(r_path, xai_model, time_sec, n_samples, n_features, iteration=None):
    """
    Function that saves the runtime of xai_models.

    :param r_path: path where to save times
    :param xai_model: name of the XAI model used to generate relevances
    :param time_sec: runtime of XAI model
    :param n_samples: number of samples, attributes were calculated for
    :param n_features: number of features, attributes were calculated for (target not included)
    :param iteration: None if the model is only validated once, Integer if the model is validated multiple times,
                      e.g. 'for iteration in range(10)'. (default = None)
    """

    create_dir(os.path.join(r_path, 'runtime'))
    # path to save file
    f_path = os.path.join(r_path, 'runtime', xai_model + '_runtime_iteration' + str(iteration) + '.csv')

    # check if the csv file should be created for multiple iterations or just one
    if iteration is not None:
        # only write the header the first iteration
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            header = ['Iteration', 'N_Samples', 'N_Features', 'Runtime']
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

        # append relevances for current iteration
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)
            # write information in row
            w.writerow([iteration, n_samples, n_features, time_sec])

    else:
        if not os.path.exists(f_path):
            # create the header for the csv file (only NN have epochs column)
            header = ['N_Samples', 'N_Features', 'Runtime']
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)
                # write information in row
                w.writerow([n_samples, n_features, time_sec])
