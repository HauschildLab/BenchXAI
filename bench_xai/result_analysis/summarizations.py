"""
    Function

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""
import csv
import os
import pickle
import cv2

import numpy.ma as ma
import pandas as pd
import numpy as np
from scipy.stats import t
from math import sqrt
from statistics import stdev
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import scipy.stats as ss


def create_metric_sum_csv(results_path, data_type, dataset_name, save_folder, metrics):
    # save path
    f_path = os.path.join(save_folder, data_type, dataset_name, 'metrics_summary.csv')
    # get path to metrics folder
    metrics_path = os.path.join(results_path, data_type, dataset_name, 'metrics')

    # get all model types (logreg, rf, pytorch)
    model_types = list(set([file.split('_')[-3] for file in os.listdir(metrics_path)]))
    # get all model metric types ('train', 'test', or 'val')
    met_types = list(set([file.split('_')[-2] for file in os.listdir(metrics_path)]))

    # create lists for headers
    first_header = [''] + [m for m in model_types for i in range(len(met_types))]
    second_header = [''] + met_types * len(model_types)

    # create the csv file with the header
    with open(f_path, 'w', newline='') as f:
        # create the csv writer
        w = csv.writer(f)
        # write the header
        w.writerow(first_header)
        w.writerow(second_header)

    # iterate over metrics
    for met in metrics:
        row = [met]
        # iterate over model_types
        for model in model_types:
            # get all metric files for model
            met_files = [file for file in os.listdir(metrics_path) if model in file]
            # iterate over metric types
            for typ in met_types:
                metrics_file = [file for file in met_files if typ in file]
                df = pd.read_csv(os.path.join(metrics_path, metrics_file[0]))
                if 'Unnamed: 0' in df.columns:
                    r = df.pop('Unnamed: 0')
                # for training just use metrics of last epoch
                if 'Epoch' in df.columns:
                    sub_df = df.loc[df['Epoch'] == max(df['Epoch'])]
                    row.append(str(np.round(np.median(sub_df[met]), 3)) + u"\u00B1" +
                               str(np.round(np.subtract(*np.percentile(sub_df[met], [75, 25])), 3)))
                else:
                    row.append(str(np.round(np.median(df[met]), 3)) + u"\u00B1" +
                               str(np.round(np.subtract(*np.percentile(df[met], [75, 25])), 3)))

        # append row for each metric
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)

            # write the header
            w.writerow(row)


def create_rel_sum_csv(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))
    # get number of unique labels
    mapping_df = p_probs.loc[p_probs['Iteration'] == 0][['IDs', 'True_Labels']]
    # iterate over unique labels/classes
    for cl in range(len(set(mapping_df['True_Labels']))):
        # save path
        f_path = os.path.join(save_folder, data_type, dataset_name, 'rel_summary_class_' + str(cl) + '.csv')
        class_ids = mapping_df.loc[mapping_df['True_Labels'] == cl]['IDs'].tolist()
        counter = 0
        # iterate over all methods
        for m in method_name:
            # init row to write
            row = [m]
            # get file
            file = os.listdir(os.path.join(rel_path, m))[0]
            # read in file
            df = pd.read_csv(os.path.join(rel_path, m, file))

            features = [x for x in df.columns.tolist() if x not in ['Iteration', 'IDs']]

            if counter == 0:
                header = ['Method'] + features
                # create the csv file with the header
                with open(f_path, 'w', newline='') as f:
                    # create the csv writer
                    w = csv.writer(f)
                    # write the header
                    w.writerow(header)

                counter += 1

            if 'IDs' in df.columns:
                class_rel = df.loc[df['IDs'].isin(class_ids)]
                class_rel.pop('IDs')
                class_rel.pop('Iteration')

                # iterate over features
                for feat in features:
                    row.append(str(np.round(np.median(class_rel[feat]), 3)) + u"\u00B1" + str(
                        np.round(np.subtract(*np.percentile(class_rel[feat], [75, 25])), 3)))
            else:
                class_rel = df
                class_rel.pop('Iteration')

                # iterate over features
                for feat in features:
                    row.append(str(np.round(np.median(class_rel[feat]), 3)) + u"\u00B1" + str(
                        np.round(np.subtract(*np.percentile(class_rel[feat], [75, 25])), 3)))

            # append row for each metric
            with open(f_path, 'a', newline='') as f:
                # create the csv writer
                w = csv.writer(f)

                # write the header
                w.writerow(row)


def create_correct_incorrect_classified_rel_sum_csv(results_path, data_type, dataset_name, save_folder, pam50=True):
    pam50_remaining = ['UBE2T', 'BIRC5', 'NUF2', 'CDC6', 'CCNB1', 'TYMS', 'MYBL2', 'CEP55', 'MELK', 'NDC80',  'RRM2',
                       'UBE2C', 'CENPF', 'PTTG1', 'EXO1', 'ORC6L', 'ANLN', 'CCNE1', 'CDC20', 'MKI67', 'KIF2C', 'ACTR3B',
                       'MYC', 'EGFR', 'KRT5', 'PHGDH', 'CDH3', 'MIA', 'KRT17', 'FOXC1', 'SFRP1', 'KRT14', 'ESR1',
                       'SLC39A6', 'BAG1', 'MAPT', 'PGR', 'CXXC5', 'MLPH', 'BCL2', 'MDM2', 'NAT1', 'FOXA1', 'BLVRA',
                       'MMP11', 'GPR160', 'FGFR4', 'GRB7', 'TMEM45B', 'ERBB2']

    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    if 'Unnamed: 0' in p_probs.columns:
        r = p_probs.pop('Unnamed: 0')

    # iterate over unique labels/classes
    for m in method_name:
        print(m)
        # get file
        file = os.listdir(os.path.join(rel_path, m))[0]
        # read in file
        if m in ['logreg', 'rf']:
            df = pd.read_csv(os.path.join(rel_path, m, file))
            if 'Unnamed: 0' in df.columns:
                r = df.pop('Unnamed: 0')
        else:
            df = pd.read_csv(os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))

        for cl in [0, 1]:
            # iterate over all methods
            for correct in [True, False]:

                # get class df
                cl_df = p_probs.loc[p_probs['True_Labels'] == cl]

                if correct:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
                else:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] < 0.5]


                # save path
                if correct:
                    f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                          'median_iqr_class_' + str(cl) + '_correctly_classified.csv')
                else:
                    f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                          'median_iqr_class_' + str(cl) + '_wrongly_classified.csv')

                if len(cl_df) > 0:
                    cl_df = cl_df[['Iteration', 'IDs']]

                    row = [m]
                    if 'IDs' in df.columns:
                        # only keep correct class relevances
                        class_rel = pd.merge(cl_df, df)
                        class_rel.pop('IDs')
                        class_rel.pop('Iteration')
                        # init row to write

                        if pam50:
                            features = pam50_remaining
                        else:
                            features = class_rel.columns.tolist()
                        
                        if not os.path.exists(f_path):
                            header = ['Method'] + features
                            # create the csv file with the header
                            with open(f_path, 'w', newline='') as f:
                                # create the csv writer
                                w = csv.writer(f)
                                # write the header
                                w.writerow(header)

                        # iterate over features
                        for feat in features:
                            row.append(str(np.round(np.median(class_rel[feat]), 3)) + u"\u00B1" + str(
                                np.round(np.subtract(*np.percentile(class_rel[feat], [75, 25])), 3)))

                    else:
                        class_rel = pd.merge(cl_df, df)
                        class_rel.pop('Iteration')

                        # iterate over features
                        for feat in features:
                            row.append(str(np.round(np.median(class_rel[feat]), 3)) + u"\u00B1" + str(
                                np.round(np.subtract(*np.percentile(class_rel[feat], [75, 25])), 3)))

                    # append row for each metric
                    with open(f_path, 'a', newline='') as f:
                        # create the csv writer
                        w = csv.writer(f)
                        # write the header
                        w.writerow(row)


def create_rel_sum_over_all_csv(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    f_path = os.path.join(save_folder, data_type, dataset_name, 'rel_summary_over_all_classes.csv')

    counter = 0

    # iterate over all methods
    for m in method_name:
        # init row to write
        row = [m]
        # get file
        file = os.listdir(os.path.join(rel_path, m))[0]
        # read in file
        df = pd.read_csv(os.path.join(rel_path, m, file))
        if 'IDs' in df.columns.tolist():
            df.pop('IDs')
        df.pop('Iteration')
        features = df.columns.tolist()

        print('Method: ', m, ', In [-0.01, 0.01]: ', ((df < 0.01) & (df > -0.01)).sum().sum(), ', Total values: ',
              10000 * len(features))

        if counter == 0:
            header = ['Method'] + features
            # create the csv file with the header
            with open(f_path, 'w', newline='') as f:
                # create the csv writer
                w = csv.writer(f)
                # write the header
                w.writerow(header)

            counter += 1

        for feat in features:
            row.append(str(np.round(np.median(df[feat]), 3)) + u"\u00B1" + str(
                np.round(np.subtract(*np.percentile(df[feat], [75, 25])), 3)))

        # append row for each metric
        with open(f_path, 'a', newline='') as f:
            # create the csv writer
            w = csv.writer(f)

            # write the header
            w.writerow(row)


def create_rel_sum_per_sample_csv(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    f_path = os.path.join(save_folder, data_type, dataset_name, 'rel_summary_per_sample_all_classes.csv')

    counter = 0

    # iterate over all methods
    for m in method_name:
        if m not in ['logreg', 'rf']:

            # get file
            file = os.listdir(os.path.join(rel_path, m))
            # read in file
            df = pd.read_csv(os.path.join(rel_path, m, file[0]))
            features = df.columns.tolist()[2:]

            if counter == 0:
                header = ['Method', 'IDs'] + features
                # create the csv file with the header
                with open(f_path, 'w', newline='') as f:
                    # create the csv writer
                    w = csv.writer(f)
                    # write the header
                    w.writerow(header)

                counter += 1

            # iterate over sample ids
            for id in df['IDs'].unique():
                # init row to write
                row = [m, id]
                id_df = df.loc[df['IDs'] == id]

                for feat in features:
                    row.append(str(np.round(np.median(id_df[feat]), 3)) + u"\u00B1" + str(
                        np.round(np.subtract(*np.percentile(id_df[feat], [75, 25])), 3)))

                # append row for each metric
                with open(f_path, 'a', newline='') as f:
                    # create the csv writer
                    w = csv.writer(f)

                    # write the header
                    w.writerow(row)


def count_neg_pos_attr_csv(results_path, data_type, dataset_name, save_folder, correct_classified_only=False,
                           wrong_classified_only=False):
    # get path to relevances folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    # iterate over unique labels/classes
    for cl in range(len(set(p_probs['True_Labels']))):
        # save path
        if correct_classified_only:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'count_neg_pos_attr_' + str(cl) + '_correctly_classified.csv')
        elif wrong_classified_only:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'count_neg_pos_attr_' + str(cl) + '_wrongly_classified.csv')
        else:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'count_neg_pos_attr_' + str(cl) + '.csv')

            # get class df
        cl_df = p_probs.loc[p_probs['True_Labels'] == cl]
        # if only correct classified
        if correct_classified_only:
            cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
            title = 'Correct classified class ' + str(cl) + ' attributions'
        elif wrong_classified_only:
            cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] < 0.5]
            title = 'Incorrect classified class ' + str(cl) + ' attributions'

        cl_df = cl_df[['Iteration', 'IDs']]

        counter = 0
        # iterate over all methods
        for m in method_name:
            # get file
            file = os.listdir(os.path.join(rel_path, m))[0]
            # read in file
            df = pd.read_csv(os.path.join(rel_path, m, file))

            features = [x for x in df.columns.tolist() if x not in ['Iteration', 'IDs']]

            if counter == 0:
                header = ['Method', 'Signum'] + features
                # create the csv file with the header
                with open(f_path, 'w', newline='') as f:
                    # create the csv writer
                    w = csv.writer(f)
                    # write the header
                    w.writerow(header)

                counter += 1

            if 'IDs' in df.columns:
                class_rel = df.loc[df['IDs'].isin(class_ids)]
                class_rel.pop('IDs')
                class_rel.pop('Iteration')

                for i in ['-1', '0', '+1']:
                    # init row to write
                    row = [m, i]
                    # iterate over features
                    for feat in features:
                        if i == '-1':
                            row.append(str(np.sum((class_rel[feat] < 0).values.ravel())))
                        elif i == '0':
                            row.append(str(np.sum((class_rel[feat] == 0).values.ravel())))
                        elif i == '+1':
                            row.append(str(np.sum((class_rel[feat] > 0).values.ravel())))

                    # append row for each metric
                    with open(f_path, 'a', newline='') as f:
                        # create the csv writer
                        w = csv.writer(f)

                        # write the header
                        w.writerow(row)

            else:
                class_rel = df
                class_rel.pop('Iteration')

                for i in ['-1', '0', '+1']:
                    # init row to write
                    row = [m, i]
                    # iterate over features
                    for feat in features:
                        if i == '-1':
                            row.append(str(np.sum((class_rel[feat] < 0).values.ravel())))
                        elif i == '0':
                            row.append(str(np.sum((class_rel[feat] == 0).values.ravel())))
                        elif i == '+1':
                            row.append(str(np.sum((class_rel[feat] > 0).values.ravel())))

                    # append row for each metric
                    with open(f_path, 'a', newline='') as f:
                        # create the csv writer
                        w = csv.writer(f)

                        # write the header
                        w.writerow(row)


def rank_norm_row_ansatz_3(row):
    # set all negative attributions to 0
    row_pos = row.copy()
    row_pos.loc[row_pos < 0] = 0
    # set all positive attributions to 0
    row_neg = row.copy()
    row_neg.loc[row_neg > 0] = 0
    # get percentiles
    pos_ranks = row_pos.rank(pct=True)
    neg_ranks = -row_neg.rank(pct=True, ascending=False)

    return neg_ranks * (row < 0) + pos_ranks * (row > 0)


def rank_norm_row_ansatz_2(row):
    # save locations of negative attributions with -1
    row_neg = row.copy()
    row_neg.loc[row_neg < 0] = -1
    row_neg.loc[row_neg >= 0] = 1

    # save locations of all zero attributions with 0
    row_zero = row.copy()
    row_zero.loc[row_zero == 0] = 0
    row_zero.loc[row_zero != 0] = 1

    # get absolute values
    row_abs = row.copy()
    row_abs = abs(row_abs)

    # get percentiles
    row_abs_rank = row_abs.rank(pct=True)
    row_final = row_abs_rank * row_zero * row_neg

    return row_final


def rank_norm(results_path, data_type, dataset_name, save_folder):
    # get path to relevances folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    if not os.path.exists(os.path.join(save_folder, data_type)):
        os.makedirs(os.path.join(save_folder, data_type))
    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name)):
        os.makedirs(os.path.join(save_folder, data_type, dataset_name))
    f_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances')
    if not os.path.exists(f_path):
        os.makedirs(f_path)

    if data_type == 'tabular_data':
        # get all xai methods and rf and log reg attributions if calculated
        method_name = [x for x in os.listdir(rel_path)]

        # iterate over all methods
        for m in method_name:
            print(m)
            # get file
            file = os.listdir(os.path.join(rel_path, m))[0]
            # read in file
            df = pd.read_csv(os.path.join(rel_path, m, file))
            if 'Unnamed: 0' in df.columns:
                r = df.pop('Unnamed: 0')
            # pop iteration column
            its = df.pop('Iteration')
            # remove id column if present
            if 'IDs' in df.columns:
                # pop id column
                ids = df.pop('IDs')
                # rank normalize each row in df
                df = df.apply(lambda row: rank_norm_row_ansatz_2(row), axis=1)
                # add ids column to front of df again
                df.insert(0, 'IDs', ids.values)

            else:
                # rank normalize each row in df
                df = df.apply(lambda row: rank_norm_row_ansatz_2(row), axis=1)

            # add iteration column to front of df again
            df.insert(0, 'Iteration', its.values)
            # save normalized df
            df.to_csv(os.path.join(f_path, file), index=False)
    elif data_type in ['image_data', 'signal_data']:
        # get path to relevances folder
        img_list = [x for x in os.listdir(rel_path)]
        # iterate over all image folders
        for img_folder in img_list:
            if not os.path.exists(os.path.join(f_path, img_folder)):
                os.makedirs(os.path.join(f_path, img_folder))
            # iterate over xai methods
            for xai in os.listdir(os.path.join(rel_path, img_folder)):
                if not os.path.exists(os.path.join(f_path, img_folder, xai)):
                    os.makedirs(os.path.join(f_path, img_folder, xai))
                # iterate over iterations
                for it_file in os.listdir(os.path.join(rel_path, img_folder, xai)):
                    # get relevances for that iteration
                    rels = pd.read_csv(os.path.join(rel_path, img_folder, xai, it_file))
                    shape = rels.shape
                    # save locations of negative attributions with -1
                    rels_neg = rels.values.flatten().copy()
                    rels_neg[rels_neg < 0] = -1
                    rels_neg[rels_neg >= 0] = 1

                    # save locations of all zero attributions with 0
                    rels_zero = rels.values.flatten().copy()
                    rels_zero[rels_zero == 0] = 0
                    rels_zero[rels_zero != 0] = 1

                    # get absolute values
                    rels_abs = rels.values.flatten().copy()
                    rels_abs = abs(rels_abs)
                    rels_abs = pd.DataFrame(rels_abs[None])  # turn into df

                    # get percentiles
                    rels_abs_rank = rels_abs.rank(axis=1, pct=True).values
                    rels_final = (rels_abs_rank * rels_zero * rels_neg).reshape(shape)
                    rels_final = pd.DataFrame(rels_final)
                    rels_final.to_csv(os.path.join(f_path, img_folder, xai, it_file), index=False)


def friedman_test(results_path, data_type, dataset_name, save_folder):
    # get path to relevances folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]
    method_name.remove('logreg')
    method_name.remove('rf')

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    l_correct = []
    l_cl = []
    l_method = []
    l_feat = []
    l_pval = []
    feats = []

    for correct_classified_only in [True, False]:
        # iterate over unique labels/classes
        for cl in range(len(set(p_probs['True_Labels']))):
            cl_df = p_probs.loc[p_probs['True_Labels'] == cl]

            if correct_classified_only:
                cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
                v = cl_df.IDs.value_counts()
                cl_df = cl_df[cl_df.IDs.isin(v.index[v.gt(99)])]

            cl_df = cl_df[['Iteration', 'IDs']]

            # iterate over all methods
            for m in method_name:
                # get file
                file = os.listdir(os.path.join(rel_path, m))[0]
                # read in file
                df = pd.read_csv(os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))
                class_rel = pd.merge(cl_df, df)

                if len(feats) == 0:
                    feats = class_rel.columns.to_list()[2:]

                for f in feats:
                    l_correct.append(correct_classified_only)
                    l_cl.append(cl)
                    l_method.append(m)
                    l_feat.append(f)
                    feat = [class_rel.loc[class_rel['Iteration'] == i][f].to_list() for i in range(100)]
                    # perform friedman test
                    p = friedmanchisquare(*feat)
                    l_pval.append(p.pvalue)

    # perform bonferroni correction
    rejected, p_adjusted, _, alpha_corrected = multipletests(l_pval, alpha=0.05,
                                                             method='bonferroni', is_sorted=False,
                                                             returnsorted=False)

    p_val_df = pd.DataFrame(
        {'Correct_classified_only': l_correct, 'Class': l_cl, 'XAI_Method': l_method, 'Feature': l_feat,
         'P_val': np.round(p_adjusted, 4)})

    for correct_classified_only in [True, False]:
        for cl in range(len(set(p_probs['True_Labels']))):
            if correct_classified_only:
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'correctly_classified_class_' + str(cl) + '_friedman_test.csv')
            else:
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'class_' + str(cl) + '_friedman_test.csv')

            sub_df = p_val_df.loc[(p_val_df['Correct_classified_only'] == correct_classified_only)
                                  & (p_val_df['Class'] == cl)]
            sub_df.pop('Correct_classified_only')
            sub_df.pop('Class')

            sub_df = sub_df.reset_index().groupby(['XAI_Method', 'Feature'])['P_val'].aggregate(
                'first').unstack().rename_axis(None, axis=1).rename_axis(None, axis=0)
            sub_df.to_csv(f_path)


def wilcoxon_test(results_path, data_type, dataset_name, save_folder):
    # get path to relevances folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]
    method_name.remove('logreg')
    method_name.remove('rf')

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    feats = []
    l_correct = []
    l_cl = []
    l_method = []
    l_iterations = []
    l_feat1 = []
    l_feat2 = []
    l_pval = []

    for correct_classified_only in [True, False]:
        # iterate over unique labels/classes
        for cl in range(len(set(p_probs['True_Labels']))):
            # get class df
            cl_df = p_probs.loc[p_probs['True_Labels'] == cl]
            # if only correct classified
            if correct_classified_only:
                cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
            cl_df = cl_df[['Iteration', 'IDs']]
            # if only correct classified
            if correct_classified_only:
                if not os.path.exists(
                        os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                     'correctly_classified_wilcoxon_test_results')):
                    os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                             'correctly_classified_wilcoxon_test_results'))
            else:
                if not os.path.exists(
                        os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                     'wilcoxon_test_results')):
                    os.makedirs(
                        os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl), 'wilcoxon_test_results'))
            # iterate over all methods
            for m in method_name:
                print(cl, m)
                # get file
                file = os.listdir(os.path.join(rel_path, m))[0]
                # read in file
                df = pd.read_csv(os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))
                class_rel = pd.merge(cl_df, df)
                class_rel.pop('IDs')

                # loop over iterations one by one
                for it in range(100):
                    it_df = class_rel.loc[class_rel['Iteration'] == it]
                    it_df.pop('Iteration')

                    if len(feats) == 0:
                        feats = it_df.columns.to_list()

                    # iterate over all features
                    for feat1 in feats:
                        for feat2 in feats:
                            l_correct.append(correct_classified_only)
                            l_cl.append(cl)
                            l_method.append(m)
                            l_iterations.append(it)
                            l_feat1.append(feat1)
                            l_feat2.append(feat2)
                            if feat1 != feat2:
                                # wilcoxon test not applicable if all differences are zero
                                if sum(it_df[feat1] - it_df[feat2] == 0) == len(it_df[feat1]):
                                    # in that case just append 0, since distributions are equal
                                    l_pval.append(None)
                                else:
                                    # if p-val < 0.05 then feat 1 > feat 2
                                    res = wilcoxon(x=it_df[feat1], y=it_df[feat2], alternative='greater')
                                    l_pval.append(res.pvalue)
                            else:
                                l_pval.append(None)

    # get indices where pvalues are None
    none_idxes = [i for i, v in enumerate(l_pval) if v is None]
    # remove None p_values
    real_p_vals = [x for x in l_pval if x is not None]

    # perform bonferroni correction
    rejected, p_adjusted, _, alpha_corrected = multipletests(real_p_vals, alpha=0.05,
                                                             method='bonferroni', is_sorted=False,
                                                             returnsorted=False)
    p_adjusted = p_adjusted.tolist()

    # now turn all significant (<0.05) p-values to 1 and the others to 0
    p_binary = [1 if i < 0.05 else 0 for i in p_adjusted]
    # re add 0 at correct indices
    for i in none_idxes:
        p_binary.insert(i, 0)

    p_val_df = pd.DataFrame({'Correct_classified_only': l_correct, 'Class': l_cl, 'XAI_Method': l_method,
                             'Iteration': l_iterations, 'Feature_1': l_feat1, 'Feature_2': l_feat2,
                             'P_val': np.round(p_binary, 4)})

    for correct_classified_only in [True, False]:
        # iterate over classes
        for cl in range(len(set(p_probs['True_Labels']))):
            # iterate over xai methods
            for xai in method_name:
                print(cl, xai)
                if correct_classified_only:
                    f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                          'correctly_classified_wilcoxon_test_results',
                                          'correctly_classified_class_' + str(
                                              cl) + '_' + xai + '_wilcoxon_test_corrected_significant_p_values_sum.csv')
                else:
                    f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                          'wilcoxon_test_results',
                                          'class_' + str(
                                              cl) + '_' + xai + '_wilcoxon_test_corrected_significant_p_values_sum.csv')

                sub_df = p_val_df.loc[(p_val_df['Correct_classified_only'] == correct_classified_only)
                                      & (p_val_df['Class'] == cl) & (p_val_df['XAI_Method'] == xai)]

                sub_df.pop('Correct_classified_only')
                sub_df.pop('Class')
                sub_df.pop('XAI_Method')
                sub_df.pop('Iteration')

                sub_df = sub_df.groupby(['Feature_1', 'Feature_2']).sum()
                sub_df = sub_df['P_val'].unstack('Feature_1').reset_index().rename_axis(None, axis=1).set_index(
                    'Feature_2').rename_axis(None, axis=0)
                sub_df.to_csv(f_path)


def median_runtimes(results_path, save_folder):
    # init dictionary
    dic = {'XAI': []}
    count = 0
    # iterate over datatype folders
    for data_type in os.listdir(results_path):

        # iterate over datasets
        for dataset in os.listdir(os.path.join(results_path, data_type)):
            if dataset not in ['ptb_xl_old']:
                runtime_folder = os.path.join(results_path, data_type, dataset, 'runtime')
                # init dataset list
                dic[dataset] = []
                # iterate over xai methods
                for xai in os.listdir(runtime_folder):
                    if count == 0:
                        # Append XAI
                        dic['XAI'].append(xai.split('_')[0])
                    # read in the dataset
                    df = pd.read_csv(os.path.join(results_path, data_type, dataset, 'runtime', xai))
                    if 'Unnamed: 0' in df.columns:
                        r = df.pop('Unnamed: 0')
                    med = np.round(np.median((1000*df['Runtime'])/df.iloc[0]['N_Samples']), 2)
                    dic[dataset].append(med)

                # update counter so that now xai methods do not get appended anymore (for the next datasets)
                count += 1

    final_df = pd.DataFrame(dic)
    final_df.to_csv(os.path.join(save_folder, 'median_runtimes.csv'), index=False)


def median_iqr_rf_logreg(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    dic = {'Model': []}
    count = 0
    # iterate over unique labels/classes
    for m in ['logreg', 'rf']:
        dic['Model'].append(m)
        # get file
        file = os.listdir(os.path.join(rel_path, m))[0]
        # read in file
        df = pd.read_csv(os.path.join(rel_path, m, file))
        # pop iteration column
        df.pop('Iteration')

        # init feature lists only once
        if count == 0:
            # iterate over features
            for feat in df.columns:
                dic[feat] = []
            count += 1

        for feat in df.columns:
            dic[feat].append(str(np.round(np.median(df[feat]), 3)) + u"\u00B1" +
                             str(np.round(np.subtract(*np.percentile(df[feat], [75, 25])), 3)))

    final_df = pd.DataFrame(dic)
    final_df.to_csv(os.path.join(save_folder, data_type, dataset_name, 'median_iqr_global_coeffs.csv'), index=False)


def count_tp_tn_fp_fn(results_path, save_folder):
    # init dictionary
    dic = {'Dataset': [], 'TP': [], 'FN': [], 'TN': [], 'FP': []}
    dic2 = {'Dataset': [], 'Class 1': [], 'Class 0': []}
    # iterate over datatype folders
    for data_type in os.listdir(results_path):

        # iterate over datasets
        for dataset_name in os.listdir(os.path.join(results_path, data_type)):
            if dataset_name not in ['ptb_xl_old']:
                dic['Dataset'].append(dataset_name)
                dic2['Dataset'].append(dataset_name)

                # get true label of samples ids
                pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
                p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
                p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))
                if 'Unnamed: 0' in p_probs.columns:
                    r = p_probs.pop('Unnamed: 0')

                # for each class only the correctly predicted samples per iteration
                cl_1_df = p_probs.loc[p_probs['True_Labels'] == 1]
                cl_0_df = p_probs.loc[p_probs['True_Labels'] == 0]

                dic['TP'].append(len(cl_1_df.loc[cl_1_df['Probabilities_Class_1'] >= 0.5]))
                dic['FN'].append(len(cl_1_df.loc[cl_1_df['Probabilities_Class_0'] >= 0.5]))
                dic['TN'].append(len(cl_0_df.loc[cl_0_df['Probabilities_Class_0'] >= 0.5]))
                dic['FP'].append(len(cl_0_df.loc[cl_0_df['Probabilities_Class_1'] >= 0.5]))

                correct_samp_counts_cl_1 = cl_1_df.loc[cl_1_df['Probabilities_Class_1'] >= 0.5][['IDs']].value_counts()
                correct_samp_counts_cl_0 = cl_0_df.loc[cl_0_df['Probabilities_Class_0'] >= 0.5][['IDs']].value_counts()

                dic2['Class 1'].append(len(correct_samp_counts_cl_1.loc[correct_samp_counts_cl_1 == 100]))
                dic2['Class 0'].append(len(correct_samp_counts_cl_0.loc[correct_samp_counts_cl_0 == 100]))

    final_df = pd.DataFrame(dic)
    final_df.to_csv(os.path.join(save_folder, 'tp_tn_fp_fn_counts.csv'), index=False)

    final_df2 = pd.DataFrame(dic2)
    final_df2.to_csv(os.path.join(save_folder, 'samples_correctly_classified_over_all_iterations.csv'), index=False)


def highest_pos_neg_counts(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    for m in method_name:
        # get file
        file = os.listdir(os.path.join(rel_path, m))[0]
        # read in file
        df = pd.read_csv(os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))

        # iterate over unique labels/classes
        for cl in range(len(set(p_probs['True_Labels']))):
            if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl))):
                os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl)))
            if not os.path.exists(
                    os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl), 'highest_lowest')):
                os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl), 'highest_lowest'))
            # if only correct classified
            for correct in [True, False]:
                print(m, cl, correct)
                # get class df
                probs_copy = p_probs.copy()
                cl_df = probs_copy.loc[probs_copy['True_Labels'] == cl]

                if correct:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
                    # save path
                    high_f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                               'highest_lowest',
                                               m + '_highest_rank_stacked_bar_class_' + str(
                                                   cl) + '_correct_classified.csv')
                    low_f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                              'highest_lowest',
                                              m + '_lowest_rank_stacked_bar_class_' + str(
                                                  cl) + '_correct_classified.csv')
                else:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] < 0.5]
                    # save path
                    high_f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                               'highest_lowest',
                                               m + '_highest_rank_stacked_bar_class_' + str(
                                                   cl) + '_wrong_classified.csv')
                    low_f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                              'highest_lowest',
                                              m + '_lowest_rank_stacked_bar_class_' + str(cl) + '_wrong_classified.csv')

                # for tcga all class 1 are correctly classified
                if len(cl_df) > 0:
                    cl_df = cl_df[['Iteration', 'IDs']]

                    if 'IDs' in df.columns:
                        copy_df = df.copy()
                        class_rel = pd.merge(cl_df, copy_df)
                        class_rel.pop('IDs')
                        class_rel.pop('Iteration')

                        count_highest_df = pd.DataFrame(class_rel.idxmax(axis="columns").value_counts()).reset_index()
                        count_lowest_df = pd.DataFrame(class_rel.idxmin(axis="columns").value_counts()).reset_index()
                        count_highest_df.columns = ['Gene', 'Count']
                        count_lowest_df.columns = ['Gene', 'Count']

                        count_highest_df.to_csv(high_f_path, index=False)
                        count_lowest_df.to_csv(low_f_path, index=False)


def count_zero(results_path, data_type, dataset_name, save_folder):
    # get path to relevances folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    # iterate over unique labels/classes
    for cl in range(len(set(p_probs['True_Labels']))):
        cl_df = p_probs.loc[p_probs['True_Labels'] == cl]
        cl_df = cl_df[['Iteration', 'IDs']]

        # iterate over all methods
        for m in method_name:
            if m not in ['logreg', 'rf']:
                # get file
                file = os.listdir(os.path.join(rel_path, m))[0]
                # read in file
                df = pd.read_csv(os.path.join(rel_path, m, file))

                # only keep correct class relevances
                class_rel = pd.merge(cl_df, df)
                class_rel.pop('IDs')
                class_rel.pop('Iteration')

                zero_feat = 0
                for feat in class_rel.columns:
                    if (class_rel[feat] == 0).all():
                        zero_feat += 1

            print(cl, m, zero_feat, class_rel.shape)


def save_all_correct_wrong_rel_file_paths(results_path, data_type, dataset_name, save_folder):
    # init overarching dictionary to save all infomation in
    dic = {}
    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))
    # get path to relevances folder
    rel_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances')
    # get path to relevances folder
    img_list = [x for x in os.listdir(rel_path)]
    # get classes
    classes = list(set([img_folder.split('_')[-1] for img_folder in img_list]))
    for cl in [0, 1]:#classes:
        dic[str(cl)] = {'All': [], 'Correct': [], 'Wrong': []}
    c = 0
    # iterate over all image folders
    for img_folder in img_list:
        print(c)
        c += 1

        if int(img_folder.split('_')[-1]) > 0:
            cl = 1
        else:
            cl = 0

        # iterate over xai methods
        for xai in os.listdir(os.path.join(rel_path, img_folder)):
            # iterate over iterations
            for it_file in os.listdir(os.path.join(rel_path, img_folder, xai)):
                # save
                dic[str(cl)]['All'].append(os.path.join(rel_path, img_folder, xai, it_file))

                temp = p_probs.loc[(p_probs['IDs'] == img_folder) & (
                        p_probs['Iteration'] == int(it_file.split('_')[-1].split('.')[0]))]
                if len(temp) != 0:
                    if temp['Probabilities_Class_' + str(cl)].values[0] >= 0.5:
                        dic[str(cl)]['Correct'].append(os.path.join(rel_path, img_folder, xai, it_file))
                    else:
                        dic[str(cl)]['Wrong'].append(os.path.join(rel_path, img_folder, xai, it_file))

    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'wb') as f:
        pickle.dump(dic, f)


def create_grouped_pos_neg_zero_table_signal_data(additional_info_path, data_type, dataset_name, save_folder):
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)
    first = True
    # iterate over classes
    for cl in list(dic.keys()):
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        for c in list(sub_dic.keys()):
            plot_dic = {'XAI': [], 'Sign': [], 'Wave': [], 'Percent': []}
            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                first = False
            if len(cl_list) != 0:
                # init counter for pos, neg, zero, total counts
                pos_sine = 0
                neg_sine = 0
                zer_sine = 0
                tot_sine = 0
                pos_squa = 0
                neg_squa = 0
                zer_squa = 0
                tot_squa = 0
                pos_base = 0
                neg_base = 0
                zer_base = 0
                tot_base = 0

                # iterate over XAI methods
                for xai in xai_methods:
                    print(xai)
                    xai_sublist = [s for s in cl_list if xai + '_' in s]

                    # get information for each xai methods
                    for path in xai_sublist:
                        rels = pd.read_csv(path)
                        sample = [sp.split('/') for sp in path.split('\\')][-3][0]

                        # try out additional infos
                        sine_feature_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                                'feature_indices', cl, sample + '.npy'))

                        sine_start_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                              'start_indices', cl, sample + '.npy'))

                        # square wave information
                        square_df = pd.read_csv(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                          'square_waves_info.csv'))
                        square_feature_idx = np.asarray(
                            eval(square_df.loc[square_df['Sample_IDs'] == sample]['Feature_IDs'].values[0]))
                        square_start_idx = np.asarray(
                            eval(square_df.loc[square_df['Sample_IDs'] == sample]['Start_IDs'].values[0]))  #

                        baseline_feature_idx = np.asarray([i for i in list(range(0, 6)) if
                                                           (i not in sine_feature_idx) and (
                                                                   i not in square_feature_idx)])

                        # get features that contain sine, square or only baseline waves
                        baseline_features = np.delete(rels,
                                                      np.append(square_feature_idx, sine_feature_idx).astype('int'),
                                                      axis=0)

                        # init baseline rels
                        baseline = []
                        if baseline_features.shape[0] != 0:
                            baseline.append(baseline_features.flatten())

                        # get relevant time frames
                        if len(sine_start_idx) != 0:
                            sine_features = np.delete(rels,
                                                      np.append(square_feature_idx, baseline_feature_idx).astype('int'),
                                                      axis=0)
                            temp = []
                            temp2 = []
                            counter = 0
                            for i in sine_start_idx:
                                # keep the 100 relevant time steps
                                temp.append(sine_features[counter, i:i + 100])
                                # remove the 100 relevant time steps
                                temp2.append(np.delete(sine_features[counter], list(range(i, i + 100))))
                                counter += 1
                            sine = np.stack(temp, axis=0).flatten()
                            sine_baseline = np.stack(temp2, axis=0).flatten()
                            baseline.append(sine_baseline)



                            # get relevant time frames
                        if len(square_start_idx) != 0:
                            square_features = np.delete(rels,
                                                        np.append(baseline_feature_idx, sine_feature_idx).astype('int'),
                                                        axis=0)
                            temp = []
                            temp2 = []
                            counter = 0
                            for i in square_start_idx:
                                temp.append(
                                    square_features[counter, i:i + 100])
                                # remove the 100 relevant time steps
                                temp2.append(np.delete(square_features[counter], list(
                                    range(i, i + 100))))
                                counter += 1
                            square = np.stack(temp, axis=0).flatten()
                            square_baseline = np.stack(temp2, axis=0).flatten()
                            baseline.append(square_baseline)

                        baseline = np.concatenate(baseline)

                        # count square wave attr
                        if len(square) > 0:
                            pos_squa += np.sum(square > 0)
                            neg_squa += np.sum(square < 0)
                            zer_squa += np.sum(square == 0)
                            tot_squa += square_features.shape[0] * square_features.shape[1]

                        # count baseline wave attr
                        if len(sine) > 0:
                            pos_sine += np.sum(sine > 0)
                            neg_sine += np.sum(sine < 0)
                            zer_sine += np.sum(sine == 0)
                            tot_sine += sine_features.shape[0] * sine_features.shape[1]

                        if baseline_features.shape[0] != 0:
                            # count baseline wave attr
                            pos_base += np.sum(baseline > 0)
                            neg_base += np.sum(baseline < 0)
                            zer_base += np.sum(baseline == 0)
                            tot_base += baseline_features.shape[0] * baseline_features.shape[1]

                    plot_dic['XAI'].extend([xai] * 9)
                    plot_dic['Wave'].extend(['Sine'] * 3)
                    plot_dic['Wave'].extend(['Base'] * 3)
                    plot_dic['Wave'].extend(['Square'] * 3)
                    plot_dic['Sign'].extend([1.0, 0.0, -1.0] * 3)
                    plot_dic['Percent'].extend([np.round((pos_sine / tot_sine) * 100, 0),
                                                np.round((zer_sine / tot_sine) * 100, 0),
                                                np.round((neg_sine / tot_sine) * 100, 0),
                                                np.round((pos_base / tot_base) * 100, 0),
                                                np.round((zer_base / tot_base) * 100, 0),
                                                np.round((neg_base / tot_base) * 100, 0),
                                                np.round((pos_squa / tot_squa) * 100, 0),
                                                np.round((zer_squa / tot_squa) * 100, 0),
                                                np.round((neg_squa / tot_squa) * 100, 0)])

                df = pd.DataFrame(plot_dic)
                df.to_csv(os.path.join(save_folder, data_type, dataset_name,
                                       'class_' + str(cl) + '_' + c + '_pos_zero_neg.csv'))


def create_median_rel_table_signal_data(additional_info_path, data_type, dataset_name, save_folder):
    plot_dic = {'XAI': [], 'Wave': [], 'Median' + u"\u00B1" + 'IQR': []}
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)

    # iterate over classes
    for cl in list(dic.keys()):
        first = True
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        # create medians over all iterations
        for c in list(sub_dic.keys()):
            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                samples = list(set([os.path.split(os.path.split(os.path.split(i)[0])[0])[-1] for i in cl_list]))
                first = False

            # iterate over XAI methods
            for xai in xai_methods:
                print(xai)
                xai_sublist = [s for s in cl_list if xai + '_' in s]

                first_sine = True
                first_baseline = True
                first_square = True
                for sample in samples:
                    sample_sublist = [s for s in xai_sublist if sample in s]
                    # counter that is only true for the first image
                    first_img = True
                    # iterate over iteration
                    for it in sample_sublist:
                        # save the first iteration image as np array
                        if first_img:
                            # load the first image as np.array
                            it_img = np.asarray(pd.read_csv(it))
                            # expand it to a 3d array
                            it_img = np.expand_dims(it_img, axis=2)
                        else:
                            # load the next image as np.array
                            temp = np.asarray(pd.read_csv(it))
                            # expand it to a 3d array
                            temp = np.expand_dims(temp, axis=2)
                            # append new image along 3rd axis
                            it_img = np.append(it_img, temp, axis=2)
                        # set first_img counter to False
                        first_img = False
                    # calculate final image as median over 3rd axis
                    rels = np.median(it_img, axis=2)

                    # try out additional infos
                    sine_feature_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                            'feature_indices', cl, sample + '.npy'))

                    sine_start_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                          'start_indices', cl, sample + '.npy'))

                    # square wave information
                    square = pd.read_csv(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                                      'square_waves_info.csv'))
                    square_feature_idx = np.asarray(
                        eval(square.loc[square['Sample_IDs'] == sample]['Feature_IDs'].values[0]))
                    square_start_idx = np.asarray(
                        eval(square.loc[square['Sample_IDs'] == sample]['Start_IDs'].values[0]))  #

                    baseline_feature_idx = np.asarray([i for i in list(range(0, 6)) if
                                                       (i not in sine_feature_idx) and (
                                                                   i not in square_feature_idx)])

                    # get features that contain sine, square or only baseline waves
                    baseline_features = np.delete(rels,
                                                  np.append(square_feature_idx, sine_feature_idx).astype('int'),
                                                  axis=0)

                    # init baseline rels
                    baseline = []
                    if baseline_features.shape[0] != 0:
                        baseline.append(baseline_features.flatten())

                    # get relevant time frames
                    if len(sine_start_idx) != 0:
                        sine_features = np.delete(rels,
                                                  np.append(square_feature_idx, baseline_feature_idx).astype('int'),
                                                  axis=0)
                        temp = []
                        temp2 = []
                        counter = 0
                        for i in sine_start_idx:
                            # keep the 100 relevant time steps
                            temp.append(sine_features[counter, i:i + 100])
                            # remove the 100 relevant time steps
                            temp2.append(np.delete(sine_features[counter], list(range(i, i + 100))))
                            counter += 1
                        sine = np.stack(temp, axis=0).flatten()
                        sine_baseline = np.stack(temp2, axis=0).flatten()
                        baseline.append(sine_baseline)

                        if first_sine:
                            sine_all_features = sine
                            first_sine = False
                        else:
                            sine_all_features = np.concatenate([sine_all_features, sine])

                        # get relevant time frames
                    if len(square_start_idx) != 0:
                        square_features = np.delete(rels,
                                                    np.append(baseline_feature_idx, sine_feature_idx).astype('int'),
                                                    axis=0)
                        temp = []
                        temp2 = []
                        counter = 0
                        for i in square_start_idx:
                            temp.append(
                                square_features[counter, i:i + 100])
                            # remove the 100 relevant time steps
                            temp2.append(np.delete(square_features[counter], list(
                                range(i, i + 100))))
                            counter += 1
                        square = np.stack(temp, axis=0).flatten()
                        square_baseline = np.stack(temp2, axis=0).flatten()
                        baseline.append(square_baseline)

                        if first_square:
                            square_all_features = square
                            first_square = False
                        else:
                            square_all_features = np.concatenate([square_all_features, square])

                    baseline = np.concatenate(baseline)

                    if first_baseline:
                        baseline_all_features = baseline
                        first_baseline = False
                    else:
                        baseline_all_features = np.concatenate([baseline_all_features, baseline])


                plot_dic['XAI'].extend([xai] * 3)
                plot_dic['Wave'].extend(['Sine'])
                plot_dic['Wave'].extend(['Base'])
                plot_dic['Wave'].extend(['Square'])
                plot_dic['Median' + u"\u00B1" + 'IQR'].extend([
                    str(np.round(np.median(sine_all_features), 3)) + u"\u00B1" +
                    str(np.round(np.subtract(*np.percentile(sine_all_features, [75, 25])), 3)),
                    str(np.round(np.median(baseline_all_features), 3)) + u"\u00B1" +
                    str(np.round(np.subtract(*np.percentile(baseline_all_features, [75, 25])), 3)),
                    str(np.round(np.median(square_all_features), 3)) + u"\u00B1" +
                    str(np.round(np.subtract(*np.percentile(square_all_features, [75, 25])), 3))])

            df = pd.DataFrame(plot_dic)
            df.to_csv(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl) + '_all_median_rels.csv'))


def create_median_rel_table_image_data(additional_info_path, data_type, dataset_name, save_folder):
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)
    # iterate over classes
    for cl in ['1']:  # list(dic.keys()):
        first = True
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        # create medians over all iterations
        for c in ['Correct']:  # list(sub_dic.keys()):
            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                samples = list(set([os.path.split(os.path.split(os.path.split(i)[0])[0])[-1] for i in cl_list]))
                first = False
            # iterate over grades
            for grade in ['2', '3', '4']:
                print(grade)
                grade_samp = [i for i in samples if 'e_' + grade in i]

                plot_dic = {'XAI': [], '1. Microaneurysms': [], '2. Haemorrhages': [], '3. Hard Exudates': [],
                            '4. Soft Exudates': [], '5. Optic Disc': [], 'Outside Mask': []}
                for xai in xai_methods:
                    rel_dic = {'1. Microaneurysms': [], '2. Haemorrhages': [], '3. Hard Exudates': [],
                               '4. Soft Exudates': [], '5. Optic Disc': [], 'Outside Mask': []}

                    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c, xai)):
                        os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c, xai))
                    print(xai)

                    xai_sublist = [s for s in cl_list if xai + '_' in s]
                    count = 0
                    for sample in grade_samp:
                        print(count)
                        count += 1
                        sample_sublist = [s for s in xai_sublist if sample in s]
                        # iterate over iteration
                        samp = sample.split('_')[0] + '_' + sample.split('_')[1]

                        # counter that is only true for the first image
                        first_img = True
                        # iterate over iteration
                        for it in sample_sublist:
                            # save the first iteration image as np array
                            if first_img:
                                # load the first image as np.array
                                it_img = np.asarray(pd.read_csv(it))
                                # expand it to a 3d array
                                it_img = np.expand_dims(it_img, axis=2)
                            else:
                                # load the next image as np.array
                                temp = np.asarray(pd.read_csv(it))
                                # expand it to a 3d array
                                temp = np.expand_dims(temp, axis=2)
                                # append new image along 3rd axis
                                it_img = np.append(it_img, temp, axis=2)
                            # set first_img counter to False
                            first_img = False
                        # calculate final image as median over 3rd axis
                        rel = np.median(it_img, axis=2)

                        # get segmentation masks
                        mas = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '1. Microaneurysms', samp + '_MA.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        mas[mas != 0] = 1
                        mas_mask = np.where((mas == 0) | (mas == 1), mas ^ 1, mas)
                        mas_values = ma.compressed(ma.masked_array(rel, mask=mas_mask))
                        rel_dic['1. Microaneurysms'].extend(list(mas_values))

                        exs = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '3. Hard Exudates', samp + '_EX.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        exs[exs != 0] = 1
                        exs_mask = np.where((exs == 0) | (exs == 1), exs ^ 1, exs)
                        exs_values = ma.compressed(ma.masked_array(rel, mask=exs_mask))
                        rel_dic['3. Hard Exudates'].extend(list(exs_values))

                        ods = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '5. Optic Disc', samp + '_OD.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        ods[ods != 0] = 1
                        ods_mask = np.where((ods == 0) | (ods == 1), ods ^ 1, ods)
                        ods_values = ma.compressed(ma.masked_array(rel, mask=ods_mask))
                        rel_dic['5. Optic Disc'].extend(list(ods_values))

                        if samp + '_HE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages')):
                            hes = cv2.imread(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages', samp + '_HE.tif'),
                                cv2.IMREAD_GRAYSCALE)
                            hes[hes != 0] = 1
                            hes_mask = np.where((hes == 0) | (hes == 1), hes ^ 1, hes)
                            hes_values = ma.compressed(ma.masked_array(rel, mask=hes_mask))
                            rel_dic['2. Haemorrhages'].extend(list(hes_values))

                        if samp + '_SE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                            ses = cv2.imread(
                                os.path.join(additional_info_path, dataset_name, '4. Soft Exudates', samp + '_SE.tif'),
                                cv2.IMREAD_GRAYSCALE)
                            ses[ses != 0] = 1
                            ses_mask = np.where((ses == 0) | (ses == 1), ses ^ 1, ses)
                            ses_values = ma.compressed(ma.masked_array(rel, mask=ses_mask))
                            rel_dic['4. Soft Exudates'].extend(list(ses_values))

                        if samp + '_HE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages')):
                            if samp + '_SE.tif' in os.listdir(
                                    os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                                normal_mask = hes + exs + mas + ods + ses
                            else:
                                normal_mask = hes + exs + mas + ods
                        else:
                            if samp + '_SE.tif' in os.listdir(
                                    os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                                normal_mask = exs + mas + ods + ses
                            else:
                                normal_mask = exs + mas + ods
                        normal_mask[normal_mask != 0] = 1
                        normal_values = ma.compressed(ma.masked_array(rel, mask=normal_mask))
                        rel_dic['Outside Mask'].extend(list(normal_values))

                    plot_dic['XAI'].append(xai)
                    plot_dic['1. Microaneurysms'].append(str(np.round(np.median(rel_dic['1. Microaneurysms']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['1. Microaneurysms'], [75, 25])), 3)))
                    plot_dic['2. Haemorrhages'].append(str(np.round(np.median(rel_dic['2. Haemorrhages']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['2. Haemorrhages'], [75, 25])), 3)))
                    plot_dic['3. Hard Exudates'].append(str(np.round(np.median(rel_dic['3. Hard Exudates']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['3. Hard Exudates'], [75, 25])), 3)))
                    plot_dic['4. Soft Exudates'].append(str(np.round(np.median(rel_dic['4. Soft Exudates']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['4. Soft Exudates'], [75, 25])), 3)))
                    plot_dic['5. Optic Disc'].append(str(np.round(np.median(rel_dic['5. Optic Disc']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['5. Optic Disc'], [75, 25])), 3)))
                    plot_dic['Outside Mask'].append(str(np.round(np.median(rel_dic['Outside Mask']), 3)) + u"\u00B1" +
                        str(np.round(np.subtract(*np.percentile(rel_dic['Outside Mask'], [75, 25])), 3)))

                    del rel_dic

                df = pd.DataFrame(plot_dic)
                df.to_csv(os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl) + '_correct_median_rels_grade_' + grade + '.csv'))


data = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/data/"
root = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/results/"
save = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/results_plots/"
metrics = ['AUROC', 'AUPRC', 'MCC']
add_info = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/additional_info/"


#root = "/home/jacqueline/Desktop/benchmark/results/"
#save = "/home/jacqueline/Desktop/benchmark/results_plots/"
#data = "/home/jacqueline/Desktop/benchmark/data/"
#add_info = "/home/jacqueline/Desktop/benchmark/additional_info/"

# save median and IQR for metrics
#create_metric_sum_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save, metrics)
#create_metric_sum_csv(root, 'tabular_data', 'heart_failure_clinical_records', save, metrics)
#create_metric_sum_csv(root, 'tabular_data', 'tcga_brca', save, metrics)
#create_metric_sum_csv(root, 'signal_data', 'synthetic_biosignal_data', save, metrics)
#create_metric_sum_csv(root, 'image_data', 'retina', save, metrics)

# save median IQR for XAI attributions
# create_rel_sum_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# create_rel_sum_csv(root, 'tabular_data', 'heart_failure_clinical_records', save)

# save median IQR for XAI attributions, but only for correctly classified examples
# create_correct_classified_rel_sum_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
#create_correct_classified_rel_sum_csv(root, 'tabular_data', 'heart_failure_clinical_records', save)

# save median and IQR for XAI attributions over all classes
# create_rel_sum_over_all_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# create_rel_sum_over_all_csv(root, 'tabular_data', 'heart_failure_clinical_records', save)

# save median and IQR for XAI attributions over all classes
# create_rel_sum_per_sample_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# create_rel_sum_per_sample_csv(root, 'tabular_data', 'heart_failure_clinical_records', save)

# count pos, neg, zero attributions
# count_neg_pos_attr_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# count_neg_pos_attr_csv(root, 'tabular_data', 'heart_failure_clinical_records', save)

# rank normalize xai methods
#rank_norm(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
#rank_norm(root, 'tabular_data', 'heart_failure_clinical_records', save)
#rank_norm(root, 'tabular_data', 'tcga_brca', save)
#rank_norm(root, 'image_data', 'retina', save)
#rank_norm(root, 'signal_data', 'synthetic_biosignal_data', save)

# friedman_test(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# friedman_test(root, 'tabular_data', 'heart_failure_clinical_records', save)

# wilcoxon_test(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# wilcoxon_test(root, 'tabular_data', 'heart_failure_clinical_records', save)

# wilcoxon_test(root, 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_classified_only=True)
# wilcoxon_test(root, 'tabular_data', 'heart_failure_clinical_records', save, correct_classified_only=True)

#median_runtimes(root, save)
#count_tp_tn_fp_fn(root, save)

# median_iqr_rf_logreg(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
# median_iqr_rf_logreg(root, 'tabular_data', 'heart_failure_clinical_records', save)

# highest_pos_neg_counts(root, 'tabular_data', 'tcga_brca', save)
# count_zero(root, 'tabular_data', 'tcga_brca', save)

#create_correct_incorrect_classified_rel_sum_csv(root, 'tabular_data', 'tcga_brca', save, pam50=False)
# create_correct_incorrect_classified_rel_sum_csv(root, 'tabular_data', 'breast_cancer_wisconsin_data', save, pam50=False)

#save_all_correct_wrong_rel_file_paths(root, 'signal_data', 'synthetic_biosignal_data', save)
#create_grouped_pos_neg_zero_table_signal_data(add_info, 'signal_data', 'synthetic_biosignal_data', save)
#create_median_rel_table_signal_data(add_info, 'signal_data', 'synthetic_biosignal_data', save)
#save_all_correct_wrong_rel_file_paths(root, 'image_data', 'retina', save)
create_median_rel_table_image_data(add_info, 'image_data', 'retina', save)