"""
    Function that creates plots of relevance heatmaps.
    The plots will show the original image alongside the heatmaps of
    all calculated XAI methods.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""
import pickle

import altair as alt
import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import itertools

from matplotlib import pyplot as plt

from bench_xai import create_dir

alt.data_transformers.enable("vegafusion")

root = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/results/"
save = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/results_plots/"
data = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/data/"
add_info = "E:/Uni/Doktor-Goettingen/Datasets/benchmark/additional_info/"


# root = "/home/jacqueline/Desktop/benchmark/results/"
# save = "/home/jacqueline/Desktop/benchmark/results_plots/"
# data = "/home/jacqueline/Desktop/benchmark/data/"
# add_info = "/home/jacqueline/Desktop/benchmark/additional_info/"


def create_normalized_relevance_heatmap_from_medians_csv(median_path, data_type, dataset_name, save_folder):
    medians = pd.read_csv(median_path, index_col=0)
    # save path
    f_path = os.path.join(save_folder, data_type, dataset_name, 'class_1',
                          'normalized_relevances_median_heatmap_flipped_class_1_correct_classified.png')
    median_dic = {'Feature': [], 'Score': [], 'Method': []}
    # iterate over all methods
    for m in medians.columns:
        if m not in ['rf', 'logreg']:
            median_dic['Feature'].extend(medians.index.tolist())
            median_dic['Method'].extend([m] * len(medians.index.tolist()))
            median_dic['Score'].extend(medians[m].to_list())
    med = pd.DataFrame(median_dic)
    chart = alt.Chart(med).mark_rect().encode(
        y='Method:O',
        x='Feature:O',
        color=alt.Color(
            'Score:Q',
            scale=alt.Scale(
                domain=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                range=['#9E1168', '#c0267e', '#dd72ad', '#f0b3d6', '#fadded', '#f5f3ef', '#e1f2ca', '#b6de87',
                       '#80bb47', '#4f9125', '#376319'],
                interpolate='rgb'
            )
        )
    ).properties(height=200, width=400).resolve_scale(color='independent')

    # save the plot
    chart.save(f_path, engine="vl-convert", ppi=300)


def create_rank_stacked_bar(results_path, min_max, data_type, dataset_name, save_folder, correct_wrong=False):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    # iterate over unique labels/classes
    for cl in range(len(set(p_probs['True_Labels']))):
        if correct_wrong:
            # save path
            if min_max == 'max':
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'highest_pos_rank_stacked_bar_class_' + str(cl) + '_correct_wrong_classified.png')
            elif min_max == 'min':
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'highest_neg_rank_stacked_bar_class_' + str(cl) + '_correct_wrong_classified.png')
        else:
            # save path
            if min_max == 'max':
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'highest_pos_rank_stacked_bar_class_' + str(cl) + '.png')
            elif min_max == 'min':
                f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                      'highest_neg_rank_stacked_bar_class_' + str(cl) + '.png')

        # if only correct classified
        if correct_wrong:
            plots_dfs = []
            subtitle = []
            for correct in [True]:  # , False]:
                # get class df
                cl_df = p_probs.loc[p_probs['True_Labels'] == cl]

                if correct:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
                else:
                    cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] < 0.5]

                if len(cl_df) > 0:
                    cl_df = cl_df[['Iteration', 'IDs']]
                    counter = 0
                    # iterate over all methods
                    for m in method_name:
                        # get file
                        file = os.listdir(os.path.join(rel_path, m))[0]
                        # read in file
                        df = pd.read_csv(
                            os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))

                        if 'IDs' in df.columns:
                            class_rel = pd.merge(cl_df, df)
                            class_rel.pop('IDs')
                            class_rel.pop('Iteration')
                            if dataset_name == 'heart_failure_clinical_records':
                                class_rel.columns = ['Age', 'A', 'CPK', 'EF', 'P', 'SC', 'SS', 'Time', 'D', 'HBP',
                                                     'Sex', 'S']

                            if counter == 0:
                                if min_max == 'max':
                                    class_rel[class_rel <= 0] = None
                                    count_df = pd.DataFrame(
                                        class_rel.idxmax(axis="columns").value_counts()).reset_index()
                                elif min_max == 'min':
                                    class_rel[class_rel >= 0] = None
                                    count_df = pd.DataFrame(
                                        class_rel.idxmin(axis="columns").value_counts()).reset_index()

                                count_df['Method'] = np.repeat(m, len(count_df))
                                count_df.columns = ['Feature', 'Count', 'Method']
                                counter += 1
                            else:
                                if min_max == 'max':
                                    temp = pd.DataFrame(class_rel.idxmax(axis="columns").value_counts()).reset_index()
                                elif min_max == 'min':
                                    temp = pd.DataFrame(class_rel.idxmin(axis="columns").value_counts()).reset_index()

                                temp['Method'] = np.repeat(m, len(temp))
                                temp.columns = ['Feature', 'Count', 'Method']

                                count_df = pd.concat([count_df, temp], ignore_index=True)

                    plots_dfs.append(count_df)

                    if correct:
                        subtitle.append('Correct classified samples')
                    else:
                        subtitle.append('Incorrect classified samples')

            if len(subtitle) == 1:
                chart = alt.Chart(plots_dfs[0]).mark_bar().encode(
                    x='Method',
                    y='Count',
                    color=alt.Color('Feature:N').scale(scheme='category20')
                ).configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
                    titleFontSize=20, labelFontSize=20, disable=False)
                chart.save(f_path, engine="vl-convert", ppi=300)

            if len(subtitle) == 2:
                plot1 = alt.Chart(plots_dfs[0], title=alt.Title(subtitle[0], fontSize=15)).mark_bar().encode(
                    x=alt.X('Method', axis=alt.Axis(ticks=False, labels=False, title=None)),
                    y='Count',
                    color=alt.Color('Feature:N').scale(scheme='category20')
                )
                plot2 = alt.Chart(plots_dfs[1], title=alt.Title(subtitle[1], fontSize=15)).mark_bar().encode(
                    x='Method',
                    y='Count',
                    color=alt.Color('Feature:N').scale(scheme='category20')
                )
                chart = (plot1 & plot2).configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
                    titleFontSize=20, labelFontSize=20, disable=True)

                chart.save(f_path, engine="vl-convert", ppi=300)

        else:
            cl_df = p_probs.loc[p_probs['True_Labels'] == cl]
            cl_df = cl_df[['Iteration', 'IDs']]

            counter = 0
            # iterate over all methods
            for m in method_name:
                # get file
                file = os.listdir(os.path.join(rel_path, m))[0]
                # read in file
                df = pd.read_csv(
                    os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))

                if 'IDs' in df.columns:
                    class_rel = pd.merge(cl_df, df)
                    class_rel.pop('IDs')
                    class_rel.pop('Iteration')

                    if counter == 0:
                        if min_max == 'max':
                            count_df = pd.DataFrame(class_rel.idxmax(axis="columns").value_counts()).reset_index()
                        elif min_max == 'min':
                            count_df = pd.DataFrame(class_rel.idxmin(axis="columns").value_counts()).reset_index()
                        count_df['Method'] = np.repeat(m, len(count_df))
                        count_df.columns = ['Feature', 'Count', 'Method']
                        counter += 1
                    else:
                        if min_max == 'max':
                            temp = pd.DataFrame(class_rel.idxmax(axis="columns").value_counts()).reset_index()
                        elif min_max == 'min':
                            temp = pd.DataFrame(class_rel.idxmin(axis="columns").value_counts()).reset_index()
                        temp['Method'] = np.repeat(m, len(temp))
                        temp.columns = ['Feature', 'Count', 'Method']

                        count_df = pd.concat([count_df, temp], ignore_index=True)

            chart = alt.Chart(count_df).mark_bar().encode(
                x='Method',
                y='Count',
                color=alt.Color('Feature:N').scale(scheme='category20')
            ).configure_axis(
                labelFontSize=15, titleFontSize=15).configure_legend(titleFontSize=20, labelFontSize=20)

            # save the plot
            chart.save(f_path, engine="vl-convert", ppi=300)


def create_rf_logreg_boxplot(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    counter = 0
    # save path
    f_path = os.path.join(save_folder, data_type, dataset_name, 'rf_logreg_boxplots_flipped.png')
    plots = [0, 0]
    # iterate over unique labels/classes
    for m in ['logreg', 'rf']:
        # get file
        file = os.listdir(os.path.join(rel_path, m))[0]
        # read in file
        df = pd.read_csv(os.path.join(rel_path, m, file))
        df.pop('Iteration')
        if 'Unnamed: 0' in df.columns:
            r = df.pop('Unnamed: 0')
        # if data_set is tcga just look at pam50 genes
        if dataset_name == 'heart_failure_clinical_records':
            df.columns = ['Age', 'A', 'CPK', 'EF', 'P', 'SC', 'SS', 'Time', 'D', 'HBP',
                          'Sex', 'S']

        if dataset_name == 'tcga_brca':
            file = open('E:/Uni/Doktor-Goettingen/Datasets/benchmark/additional_info/tcga_brca/pam50.txt', 'r')
            content = file.read()
            df = df[content.split(', ')]
            file.close()

        if m == 'logreg':
            if dataset_name == 'heart_failure_clinical_records':
                df = np.exp(df)
                plots[counter] = alt.Chart(df, title=alt.Title('Logistic Regression')).transform_fold(
                    df.columns.to_list(),
                    as_=['Feature', 'Global Coefficients']
                ).mark_boxplot().encode(
                    alt.X("Feature:N"),
                    alt.Y("Global Coefficients:Q").scale(zero=False, type="log"),
                    alt.Color("Feature:N").scale(scheme='category20').legend(None)
                ).properties(height=200, width=200)
                counter += 1
            else:
                df = np.exp(df)
                plots[counter] = alt.Chart(df, title=alt.Title('Logistic Regression')).transform_fold(
                    df.columns.to_list(),
                    as_=['Feature', 'Global Coefficients']
                ).mark_boxplot(size=6).encode(
                    alt.Y("Feature:N"),
                    alt.X("Global Coefficients:Q").scale(zero=False),
                    alt.Color("Feature:N").scale(scheme='category20').legend(None)
                ).properties(height=400, width=200)
                counter += 1
        else:
            plots[counter] = alt.Chart(df, title=alt.Title('Random Forest')).transform_fold(
                df.columns.to_list(),
                as_=['Feature', 'Global Coefficients']
            ).mark_boxplot(size=6).encode(
                alt.Y("Feature:N"),
                alt.X("Global Coefficients:Q").scale(zero=False),
                alt.Color("Feature:N").scale(scheme='category20').legend(None)
            ).properties(height=400, width=200)

    chart = (plots[0] | plots[1])
    # save the plot
    chart.save(f_path, engine="vl-convert", ppi=300)


def create_pos_neg_zero_stacked_bar(results_path, data_type, dataset_name, save_folder, correct_classified_only=False,
                                    wrong_classified_only=False):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    # iterate over unique labels/classes
    for cl in range(len(set(p_probs['True_Labels']))):
        # init list for plots
        plots = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # save path
        if correct_classified_only:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'pos_neg_zero_counts_stacked_bar_class_' + str(
                                      cl) + '_correctly_classified_subset.png')
        elif wrong_classified_only:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'pos_neg_zero_counts_stacked_bar_class_' + str(cl) + '_wrongly_classified.png')
        else:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                  'pos_neg_zero_counts_stacked_bar_class_' + str(cl) + '.png')

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
        for m in ['IntegratedGradients', 'DeepLift', 'Lime', 'KernelShap']:  # method_name:
            if m not in ['logreg', 'rf']:
                final_dic = {'Feature': [], 'Area': [], 'Counts': []}
                # get file
                file = os.listdir(os.path.join(rel_path, m))[0]
                # read in file
                df = pd.read_csv(os.path.join(rel_path, m, file))

                # only keep correct class relevances
                class_rel = pd.merge(cl_df, df)
                class_rel.pop('IDs')
                class_rel.pop('Iteration')
                if dataset_name == 'heart_failure_clinical_records':
                    class_rel.columns = ['Age', 'A', 'CPK', 'EF', 'P', 'SC', 'SS', 'Time', 'D', 'HBP', 'Sex', 'S']

                for feat in class_rel.columns:
                    # count sine wave attr
                    pos1 = np.sum(class_rel[feat] > 0.5)
                    neg1 = np.sum(class_rel[feat] < -0.5)
                    pos2 = np.sum((class_rel[feat] > 0) & (class_rel[feat] <= 0.5))
                    neg2 = np.sum((class_rel[feat] < 0) & (class_rel[feat] >= -0.5))
                    zero = np.sum(class_rel[feat] == 0)

                    final_dic['Feature'].extend([feat] * 5)
                    final_dic['Area'].extend(['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)'])
                    final_dic['Counts'].extend([pos1, pos2, zero, neg2, neg1])

                final_df = pd.DataFrame(final_dic)
                order = ['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)']
                plots[counter] = alt.Chart(final_df, title=alt.Title(m, fontSize=20)).transform_calculate(
                    order=f"-indexof({order}, datum.Area)"
                ).mark_bar().encode(
                    x='Feature:O',
                    y='Counts',
                    color=alt.Color('Area:N', scale=alt.Scale(domain=order,
                                                              range=['#376319', '#4f9125', '#f5f3ef', '#c0267e',
                                                                     '#8a1959'])),
                    order="order:Q"
                ).properties(
                    height=200
                )
                counter += 1

        #chart = ((plots[0] & plots[1] & plots[2]) | (plots[3] & plots[4] & plots[5]) | (
        #        plots[6] & plots[7] & plots[8]) | (plots[9] & plots[10] & plots[11]) | (
        #                 plots[12] & plots[13] & plots[14])
        #         ).properties(title=alt.Title(title, fontSize=20)).configure_axis(labelFontSize=15,
        #                                                                          titleFontSize=15).configure_legend(
        #    titleFontSize=15, labelFontSize=15)
        chart = ((plots[0] | plots[1]) & (plots[2] | plots[3])). \
            properties(). \
            configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(
            titleFontSize=15, labelFontSize=15)

        # save the plot
        chart.save(f_path, engine="vl-convert", ppi=300)


def create_stacked_density_plot(results_path, data_type, dataset_name, save_folder, correct_classified_only=False,
                                wrong_classified_only=False):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')

    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    # get true label of samples ids
    pred_probs_path = os.path.join(results_path, data_type, dataset_name, 'pred_probs')
    p_file = os.listdir(os.path.join(results_path, data_type, dataset_name, 'pred_probs'))[0]
    p_probs = pd.read_csv(os.path.join(pred_probs_path, p_file))

    # iterate over unique labels/classes
    for cl in range(len(set(p_probs['True_Labels']))):
        # init list for plots
        plots = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # save path
        f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl))

        # get class df
        cl_df = p_probs.loc[p_probs['True_Labels'] == cl]
        # if only correct classified
        if correct_classified_only:
            cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] >= 0.5]
        elif wrong_classified_only:
            cl_df = cl_df.loc[cl_df['Probabilities_Class_' + str(cl)] < 0.5]

        cl_df = cl_df[['Iteration', 'IDs']]

        counter = 0
        # iterate over all methods
        for m in ['IntegratedGradients', 'DeepLift', 'Lime', 'KernelShap']:  # method_name:
            # get file
            file = os.listdir(os.path.join(rel_path, m))[0]
            # read in file
            # df = pd.read_csv(os.path.join(rel_path, m, file))
            df = pd.read_csv(os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', file))
            if 'IDs' in df.columns.to_list():
                # only keep correct class relevances
                class_rel = pd.merge(cl_df, df)
                class_rel.pop('IDs')
                class_rel.pop('Iteration')
                if dataset_name == 'heart_failure_clinical_records':
                    class_rel.columns = ['Age', 'A', 'CPK', 'EF', 'P', 'SC', 'SS', 'Time', 'D', 'HBP', 'Sex', 'S']

                chart = alt.Chart(class_rel, title=alt.Title(m, fontSize=20)).transform_fold(
                        class_rel.columns.to_list(),
                        as_=['Feature', 'value']
                    ).transform_density(
                        density='value',
                        bandwidth=0.3,
                        groupby=['Feature'],
                        steps=200
                    ).mark_area(opacity=0.5, clip=True).encode(
                        alt.X('value:Q').scale(domain=[-1, 1]),
                        alt.Y('density:Q').stack(None),
                        alt.Color('Feature:N', scale=alt.Scale(scheme='category20'))
                    ).properties(height=200, width=200)

                plots[counter] = chart  # + rules

                counter += 1

        chart = ((plots[0] | plots[1]) & (plots[2] | plots[3])) \
            .configure_axis(labelFontSize=15,
                            titleFontSize=15) \
            .configure_legend(titleFontSize=20,
                              labelFontSize=20)


        #chart = ((plots[0] & plots[1] & plots[2]) | (plots[3] & plots[4] & plots[5]) | (
        #        plots[6] & plots[7] & plots[8]) | (plots[9] & plots[10] & plots[11]) | (
        #                 plots[12] & plots[13] & plots[14])
        #         ).configure_axis(labelFontSize=15, titleFontSize=15).configure_legend(titleFontSize=20,
        #                                                                               labelFontSize=20)

        if correct_classified_only:
            #chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density_correctly_classified.png'), engine="vl-convert", ppi=300)
            chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density_correctly_classified_subset.png'))
        elif wrong_classified_only:
            #chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density_wrongly_classified.png'), engine="vl-convert", ppi=300)
            chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density_wrongly_classified_subset.png'))
        else:
            # chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density.png'), engine="vl-convert", ppi=300)
            chart.save(os.path.join(f_path, 'class_' + str(cl) + '_rank_norm_density_subset.png'))


def create_wilcoxon_sums_heatmap(results_path, data_type, dataset_name, save_folder):
    # get path to metrics folder
    rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
    # get all xai methods and rf and log reg attributions if calculated
    method_name = [x for x in os.listdir(rel_path)]

    for correct_classified_only in [True]:
        # iterate over unique labels/classes
        for cl in [1]:
            f_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl))
            # init list for plots
            plots = []

            # save path
            if correct_classified_only:
                save_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                         'correctly_classified_wilcoxon_test_results')
            else:
                save_path = os.path.join(save_folder, data_type, dataset_name, 'class_' + str(cl),
                                         'wilcoxon_test_results')

            # iterate over all methods
            for m in ['IntegratedGradients', 'DeepLift', 'Lime', 'KernelShap']:  # method_name:
                if m not in ['logreg', 'rf']:
                    if correct_classified_only:
                        f = 'correctly_classified_class_' + str(
                            cl) + '_' + m + '_wilcoxon_test_corrected_significant_p_values_sum.csv'
                    else:
                        f = 'class_' + str(cl) + '_' + m + '_wilcoxon_test_corrected_significant_p_values_sum.csv'
                    # get file
                    file = os.path.join(save_path, f)
                    # read in file
                    df = pd.read_csv(file, index_col=0)
                    # init dic for data frame
                    d = {'Feature 1': [], 'Feature 2': [], 'Score': []}

                    if dataset_name == 'heart_failure_clinical_records':
                        df.columns = ['Age', 'A', 'CPK', 'D', 'EF', 'HBP', 'P', 'SC', 'SS', 'Sex', 'S', 'Time']
                        df.index = ['Age', 'A', 'CPK', 'D', 'EF', 'HBP', 'P', 'SC', 'SS', 'Sex', 'S', 'Time']

                    for f1 in df.columns.to_list():
                        d['Feature 1'] = d['Feature 1'] + [f1] * len(df.columns.to_list())
                        d['Score'] = d['Score'] + df[f1].values.tolist()
                        d['Feature 2'] = d['Feature 2'] + list(df.index.values)

                    total_df = pd.DataFrame(d)
                    chart = alt.Chart(total_df, title=alt.Title(m, fontSize=20)).mark_rect().encode(
                        x='Feature 1:O',
                        y='Feature 2:O',
                        color=alt.Color(
                            'Score:Q',
                            bin=True,
                            scale=alt.Scale(
                                bins=[0, 20, 40, 60, 80, 100],
                                range=['#f5f3ef', '#e1f2ca', '#b6de87', '#80bb47', '#4f9125', '#376319']
                            )
                        )
                    ).resolve_scale(color='independent').properties(height=200, width=200)

                    plots.append(chart)

            #final_chart = ((plots[0] & plots[1] & plots[2]) | (plots[3] & plots[4] & plots[5]) | (
            #        plots[6] & plots[7] & plots[8]) | (plots[9] & plots[10] & plots[11]) | (
            #                       plots[12] & plots[13] & plots[14])).configure_axis(labelFontSize=15,
            #                                                                          titleFontSize=15).configure_legend(
            #    titleFontSize=20, labelFontSize=20)

            final_chart = ((plots[0] | plots[1]) & (plots[2] | plots[3])).configure_axis(labelFontSize=15,
                                                                                                titleFontSize=15).configure_legend(
                titleFontSize=20, labelFontSize=20)

            if correct_classified_only:
                final_chart.save(os.path.join(f_path, 'correctly_classified_class_' + str(cl) + '_wilcoxon_subset.png'))
            else:
                final_chart.save(os.path.join(f_path, 'class_' + str(cl) + '_wilcoxon.svg'))


def create_img_rel_plots(data_path, data_type, dataset_name, image, save_folder):
    plt.rcParams.update({'font.size': 25})

    img_path = os.path.join(data_path, data_type, dataset_name, '1', image)
    rel_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances', image)

    names = ["Original"]
    orig_img = cv2.imread(img_path + '.jpg')
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    all_images_list = [orig_img]
    # iterate over each xai relevance method for this image
    for xai in ['IntegratedGradients', 'DeepLift', 'LRP-Alpha1-Beta0', 'Occlusion',
                'GuidedBackprop']:  # os.listdir(rel_path):

        names.append(xai)
        xai_method_path = os.path.join(rel_path, xai)
        # check how often the image was in the validation dataset
        xai_iterations = os.listdir(xai_method_path)

        # counter that is only true for the first image
        first_img = True
        for it in xai_iterations:
            # save the first iteration image as np array
            if first_img:
                # load the first image as np.array
                it_img = np.asarray(pd.read_csv(os.path.join(xai_method_path, it)))
                # expand it to a 3d array
                it_img = np.expand_dims(it_img, axis=2)
            else:
                # load the next image as np.array
                temp = np.asarray(pd.read_csv(os.path.join(xai_method_path, it)))
                # expand it to a 3d array
                temp = np.expand_dims(temp, axis=2)
                # append new image along 3rd axis
                it_img = np.append(it_img, temp, axis=2)
            # set first_img counter to False
            first_img = False

        # calculate final image as median over 3rd axis
        median_img = np.median(it_img, axis=2)

        final_image = pd.DataFrame(median_img)
        all_images_list.append(final_image)

    fig, axs = plt.subplots(2, 3, figsize=(25, 10))

    # fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.tight_layout()
    axs = axs.ravel()

    for i in range(len(all_images_list)):
        if i == 0:
            img = axs[i].imshow(all_images_list[i])
        else:
            img = axs[i].imshow(all_images_list[i], vmin=-1, vmax=1, cmap="PiYG")

        # set subplot title
        axs[i].set_title(names[i])
        axs[i].axis('off')

    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(img, cax=cbar_ax)

    plt.savefig(os.path.join(save_folder, data_type, dataset_name, image + '_relevances_subset.jpg'))
    plt.close(fig)


def create_sig_rel_plots(results_path, data_path, data_type, dataset_name, save_folder, norm=True):
    if norm:
        # get path to relevances folder
        rel_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances')
        folder = 'norm_rel_plots'
    else:
        # get path to relevances folder
        rel_path = os.path.join(results_path, data_type, dataset_name, 'relevances')
        folder = 'raw_rel_plots'
    # create folder for saving plots
    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, folder)):
        os.makedirs(os.path.join(save_folder, data_type, dataset_name, folder))

    # get path to relevances folder
    signal_path = os.path.join(data_path, data_type, dataset_name)
    # get path to relevances folder
    img_list = [x for x in os.listdir(rel_path)]
    # iterate over all image folders
    for img_folder in img_list:
        # create folder for saving plots
        if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, folder, img_folder)):
            os.makedirs(os.path.join(save_folder, data_type, dataset_name, folder, img_folder))

        sig = np.load(os.path.join(signal_path, img_folder.split('_')[1], img_folder + '.npy'))
        # iterate over xai methods
        for xai in os.listdir(os.path.join(rel_path, img_folder)):
            # create folder for saving plots
            if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, folder, img_folder, xai)):
                os.makedirs(os.path.join(save_folder, data_type, dataset_name, folder, img_folder, xai))
            # iterate over iterations
            for it_file in os.listdir(os.path.join(rel_path, img_folder, xai)):
                # get relevances for that iteration
                rels = pd.read_csv(os.path.join(rel_path, img_folder, xai, it_file)).to_numpy()
                steps = list(range(sig.shape[1])) * sig.shape[0]
                sample_dic = {'Steps': steps, 'Feature': [], 'Values': [], 'Explanations': []}

                for channel in range(sig.shape[0]):
                    sample_dic['Feature'].extend(['Feature ' + str(channel + 1)] * sig.shape[1])
                    sample_dic['Values'].extend(sig[channel, :].tolist())
                    sample_dic['Explanations'].extend(rels[channel, :].tolist())

                sample_df = pd.DataFrame.from_dict(sample_dic)

                sa = alt.Chart(sample_df).mark_line().encode(
                    x='Steps:Q',
                    y='Values:Q',
                    color=alt.value("#000000")
                )

                if norm:
                    xaic = alt.Chart(sample_df).mark_rect(width=2).encode(
                        x="Steps",
                        color=alt.Color('Explanations', scale=alt.Scale(scheme='purplegreen', domain=[-1, 1])),

                    )
                else:
                    xaic = alt.Chart(sample_df).mark_rect(width=2).encode(
                        x="Steps",
                        color=alt.Color('Explanations', scale=alt.Scale(scheme='purplegreen')),

                    )

                chart = alt.layer(xaic, sa, data=sample_df).properties(
                    width=10000,
                    height=100
                ).facet(
                    facet=alt.Facet("Feature:N", header=alt.Header(
                        title=xai + ' relevances for class ' + img_folder.split('_')[1] + ' sample ' +
                              img_folder.split('_')[-1] + ' iteration ' + it_file.split('_')[-1].split('.')[0],
                        titleFontSize=20, labelFontSize=15)),
                    columns=1
                ).configure_axis(
                    labelFontSize=15,
                    titleFontSize=15
                ).configure_legend(
                    titleFontSize=15,
                    labelFontSize=15
                )

                chart.save(os.path.join(save_folder, data_type, dataset_name, folder, img_folder, xai,
                                        it_file.split('.')[0] + '.svg'))


def create_median_sig_rel_plots(data_path, data_type, dataset_name, save_folder):
    # create folder for saving plots
    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all')):
        os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all'))
    # get path to relevances folder
    rel_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances')
    # get path to relevances folder
    signal_path = os.path.join(data_path, data_type, dataset_name)
    # get path to relevances folder
    img_list = [x for x in os.listdir(rel_path)]
    # iterate over all image folders
    for img_folder in img_list:
        # create folder for saving plots
        if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all', img_folder)):
            os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all', img_folder))
        # iterate over xai methods
        for xai in os.listdir(os.path.join(rel_path, img_folder)):
            # create folder for saving plots
            if not os.path.exists(
                    os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all', img_folder, xai)):
                os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all', img_folder, xai))
            # iterate over iterations
            # counter that is only true for the first image
            first_img = True
            for it_file in os.listdir(os.path.join(rel_path, img_folder, xai)):
                if first_img:
                    # load the first image as np.array
                    it_img = np.asarray(pd.read_csv(os.path.join(rel_path, img_folder, xai, it_file)))
                    # expand it to a 3d array
                    it_img = np.expand_dims(it_img, axis=2)
                    # load original signal
                    sig = np.load(os.path.join(signal_path, img_folder.split('_')[1], img_folder + '.npy'))
                    # set first_img counter to False
                    first_img = False
                else:
                    # load the next image as np.array
                    temp = np.asarray(pd.read_csv(os.path.join(rel_path, img_folder, xai, it_file)))
                    # expand it to a 3d array
                    temp = np.expand_dims(temp, axis=2)
                    # append new image along 3rd axis
                    it_img = np.append(it_img, temp, axis=2)

            # calculate final image as median over 3rd axis
            median_img = np.median(it_img, axis=2)

            steps = list(range(sig.shape[1])) * sig.shape[0]
            sample_dic = {'Steps': steps, 'Feature': [], 'Values': [], 'Explanations': []}

            for channel in range(sig.shape[0]):
                sample_dic['Feature'].extend(['Feature ' + str(channel + 1)] * sig.shape[1])
                sample_dic['Values'].extend(sig[channel, :].tolist())
                sample_dic['Explanations'].extend(median_img[channel, :].tolist())

            sample_df = pd.DataFrame.from_dict(sample_dic)

            sa = alt.Chart(sample_df).mark_line().encode(
                x='Steps:Q',
                y='Values:Q',
                color=alt.value("#000000")
            )

            xaic = alt.Chart(sample_df).mark_rect(width=2).encode(
                x="Steps",
                color=alt.Color('Explanations', scale=alt.Scale(scheme='purplegreen', domain=[-1, 1])),

            )

            chart = alt.layer(xaic, sa, data=sample_df).properties(
                width=1000,
                height=100
            ).facet(
                facet=alt.Facet("Feature:N", header=alt.Header(
                    title='Median ' + xai + ' relevances for class ' + img_folder.split('_')[1] + ' sample ' +
                          img_folder.split('_')[-1], titleFontSize=20, labelFontSize=15)),
                columns=1
            ).configure_axis(
                labelFontSize=15,
                titleFontSize=15
            ).configure_legend(
                titleFontSize=15,
                labelFontSize=15
            )

            chart.save(os.path.join(save_folder, data_type, dataset_name, 'median_rel_plots_all', img_folder, xai,
                                    xai + '_median_relevance.svg'))


def create_median_sig_rel_plot_of_only_sine_and_square_waves(img, additional_info_path, data_path, data_type,
                                                             dataset_name, save_folder):
    # create folder for saving example plots
    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'local_median_rel_plots', img)):
        os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'local_median_rel_plots', img))
    # get path to relevances folder
    rel_path = os.path.join(save_folder, data_type, dataset_name, 'rank_normalized_relevances')
    # get path to relevances folder
    signal_path = os.path.join(data_path, data_type, dataset_name)
    # load original signal
    sig = np.load(os.path.join(signal_path, img.split('_')[1], img + '.npy'))

    # try out additional infos
    sine_feature_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                            'feature_indices', img.split('_')[1], img + '.npy'))

    sine_start_idx = np.load(os.path.join(additional_info_path, 'synthetic_biosignal_data',
                                          'start_indices', img.split('_')[1], img + '.npy'))
    # square wave information
    square = pd.read_csv(os.path.join(additional_info_path, 'synthetic_biosignal_data', 'square_waves_info.csv'))
    square_feature_idx = np.asarray(eval(square.loc[square['Sample_IDs'] == img]['Feature_IDs'].values[0]))
    square_start_idx = np.asarray(eval(square.loc[square['Sample_IDs'] == img]['Start_IDs'].values[0]))

    first_subset = True
    for i in range(len(sine_feature_idx)):
        if first_subset:
            subset_sig = np.expand_dims(sig[sine_feature_idx[i], sine_start_idx[i]:sine_start_idx[i] + 100], axis=0)
            first_subset = False
        else:
            temp = np.expand_dims(sig[sine_feature_idx[i], sine_start_idx[i]:sine_start_idx[i] + 100], axis=0)
            subset_sig = np.append(subset_sig, temp, axis=0)

    for i in range(len(square_feature_idx)):
        temp = np.expand_dims(sig[square_feature_idx[i], square_start_idx[i]:square_start_idx[i] + 100], axis=0)
        subset_sig = np.append(subset_sig, temp, axis=0)

    # get feature number list
    feats = list(sine_feature_idx) + list(square_feature_idx)
    plots = []
    # iterate over xai methods
    for xai in ['IntegratedGradients', 'DeepLift', 'Deconvolution', 'GuidedBackprop']:  # os.listdir(os.path.join(rel_path, img)):
        # iterate over iterations
        # counter that is only true for the first image
        first_img = True
        for it_file in os.listdir(os.path.join(rel_path, img, xai)):
            if first_img:
                # load the first image as np.array
                it_img = np.asarray(pd.read_csv(os.path.join(rel_path, img, xai, it_file)))
                # expand it to a 3d array
                it_img = np.expand_dims(it_img, axis=2)
                # set first_img counter to False
                first_img = False
            else:
                # load the next image as np.array
                temp = np.asarray(pd.read_csv(os.path.join(rel_path, img, xai, it_file)))
                # expand it to a 3d array
                temp = np.expand_dims(temp, axis=2)
                # append new image along 3rd axis
                it_img = np.append(it_img, temp, axis=2)

        # calculate final image as median over 3rd axis
        median_img = np.median(it_img, axis=2)

        first_subset = True
        for i in range(len(sine_feature_idx)):
            if first_subset:
                subset_rel = np.expand_dims(median_img[sine_feature_idx[i], sine_start_idx[i]:sine_start_idx[i] + 100],
                                            axis=0)
                first_subset = False
            else:
                temp = np.expand_dims(median_img[sine_feature_idx[i], sine_start_idx[i]:sine_start_idx[i] + 100],
                                      axis=0)
                subset_rel = np.append(subset_rel, temp, axis=0)

        for i in range(len(square_feature_idx)):
            temp = np.expand_dims(median_img[square_feature_idx[i], square_start_idx[i]:square_start_idx[i] + 100],
                                  axis=0)
            subset_rel = np.append(subset_rel, temp, axis=0)

        steps = list(range(subset_sig.shape[1])) * subset_sig.shape[0]
        sample_dic = {'Steps': steps, 'Feature': [], 'Values': [], 'Relevance': []}

        counter = 0
        for f in feats:
            sample_dic['Feature'].extend(['Feature ' + str(f + 1)] * subset_sig.shape[1])
            sample_dic['Values'].extend(subset_sig[counter, :].tolist())
            sample_dic['Relevance'].extend(subset_rel[counter, :].tolist())
            counter += 1

        sample_df = pd.DataFrame.from_dict(sample_dic)

        sa = alt.Chart(sample_df).mark_line().encode(
            x='Steps:Q',
            y='Values:Q',
            color=alt.value("#000000")
        )

        xaic = alt.Chart(sample_df).mark_rect(width=2).encode(
            x="Steps",
            color=alt.Color('Relevance', scale=alt.Scale(scheme='purplegreen', domain=[-1, 1])),

        )

        chart = alt.layer(xaic, sa, data=sample_df).properties(
            width=150,
            height=100
        ).facet(
            facet=alt.Facet("Feature:N", header=alt.Header(
                title=xai, titleFontSize=20, labelFontSize=15)),
            columns=5
        )

        plots.append(chart)

        # chart.save(os.path.join(save_folder, data_type, dataset_name, 'local_median_rel_plots', img,
        #                        xai + '_local_median_relevance.png'))

    final_chart = (plots[0] & plots[1] & plots[2] & plots[3]).configure_axis(
        labelFontSize=15, titleFontSize=15).configure_legend(
        titleFontSize=15, labelFontSize=15)

    final_chart.save(os.path.join(save_folder, data_type, dataset_name, 'local_median_rel_plots', img,
                                  'subset_local_median_relevance.png'))


def plot_example_signal(img, data_path, data_type, dataset_name, save_folder):
    # get path to relevances folder
    signal_path = os.path.join(data_path, data_type, dataset_name)
    # load original signal
    sig = np.load(os.path.join(signal_path, img.split('_')[1], img + '.npy'))

    steps = list(range(sig.shape[1])) * sig.shape[0]
    sample_dic = {'Steps': steps, 'Feature': [], 'Values': []}

    for channel in range(sig.shape[0]):
        sample_dic['Feature'].extend(['Feature ' + str(channel + 1)] * sig.shape[1])
        sample_dic['Values'].extend(sig[channel, :].tolist())

    sample_df = pd.DataFrame.from_dict(sample_dic)

    sa = alt.Chart(sample_df).mark_line().encode(
        x='Steps:Q',
        y='Values:Q',
        color=alt.value("#000000")
    ).properties(
        width=400,
        height=70
    ).facet(facet=alt.Facet("Feature:N", header=alt.Header(labelFontSize=15)), columns=1
            ).configure_axis(
        labelFontSize=15,
        titleFontSize=15
    ).configure_legend(
        titleFontSize=15,
        labelFontSize=15
    )

    sa.save(os.path.join(save_folder, data_type, dataset_name, 'example_signal.png'))


def create_grouped_pos_neg_zero_stacked_bar_signal_data(additional_info_path, data_type, dataset_name, save_folder):
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)
    first = True
    # iterate over classes
    for cl in ['1']:  # list(dic.keys()):
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        for c in ['Correct']:  # list(sub_dic.keys()):

            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                first = False
            if len(cl_list) != 0:

                plots = []
                # iterate over XAI methods
                for xai in ['IntegratedGradients', 'DeepLift', 'Deconvolution', 'GuidedBackprop']:  # xai_methods:
                    print(xai)
                    # init counter for pos, neg, zero, total counts
                    pos1_sine = 0
                    neg1_sine = 0
                    pos2_sine = 0
                    neg2_sine = 0
                    zer_sine = 0
                    tot_sine = 0
                    pos1_squa = 0
                    neg1_squa = 0
                    pos2_squa = 0
                    neg2_squa = 0
                    zer_squa = 0
                    tot_squa = 0
                    pos1_base = 0
                    neg1_base = 0
                    pos2_base = 0
                    neg2_base = 0
                    zer_base = 0
                    tot_base = 0

                    plot_dic = {'XAI': [], 'Area': [], 'Wave': [], 'Percent': []}
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
                        sine_features = np.delete(rels,
                                                  np.append(square_feature_idx, baseline_feature_idx).astype('int'),
                                                  axis=0)
                        square_features = np.delete(rels,
                                                    np.append(baseline_feature_idx, sine_feature_idx).astype('int'),
                                                    axis=0)

                        # get relevant time frames
                        if len(sine_start_idx) != 0:
                            temp = []
                            temp2 = []
                            counter = 0
                            for i in sine_start_idx:
                                # keep the 100 relevant time steps
                                temp.append(sine_features[counter, i:i + 100])
                                # remove the 100 relevant time steps
                                temp2.append(np.delete(sine_features[counter], list(range(i, i + 100))))
                                counter += 1
                            sine = np.stack(temp, axis=0)
                            sine_baseline = np.stack(temp2, axis=0)

                            # count sine wave attr
                            pos1_sine += np.sum(sine > 0.5)
                            neg1_sine += np.sum(sine < -0.5)
                            pos2_sine += np.sum((sine > 0) & (sine <= 0.5))
                            neg2_sine += np.sum((sine < 0) & (sine >= -0.5))
                            zer_sine += np.sum(sine == 0)
                            tot_sine += sine.shape[0] * sine.shape[1]

                            # count baseline wave attr
                            pos1_base += np.sum(sine_baseline > 0.5)
                            neg1_base += np.sum(sine_baseline < -0.5)
                            pos2_base += np.sum((sine_baseline > 0) & (sine_baseline <= 0.5))
                            neg2_base += np.sum((sine_baseline < 0) & (sine_baseline >= -0.5))
                            zer_base += np.sum(sine_baseline == 0)
                            tot_base += sine_baseline.shape[0] * sine_baseline.shape[1]

                            # get relevant time frames
                        if len(square_start_idx) != 0:
                            temp = []
                            temp2 = []
                            counter = 0
                            for i in square_start_idx:
                                temp.append(
                                    square_features[counter, i:i + 100])
                                # remove the 100 relevant time steps
                                temp2.append(np.delete(square_features[counter], list(range(i, i + 100))))
                                counter += 1
                            square = np.stack(temp, axis=0)
                            square_baseline = np.stack(temp2, axis=0)

                            # count square wave attr
                            pos1_squa += np.sum(square > 0.5)
                            neg1_squa += np.sum(square < -0.5)
                            pos2_squa += np.sum((square > 0) & (square <= 0.5))
                            neg2_squa += np.sum((square < 0) & (square >= -0.5))
                            zer_squa += np.sum(square == 0)
                            tot_squa += square.shape[0] * square.shape[1]

                            # count baseline wave attr
                            pos1_base += np.sum(square_baseline > 0.5)
                            neg1_base += np.sum(square_baseline < -0.5)
                            pos2_base += np.sum((square_baseline > 0) & (square_baseline <= 0.5))
                            neg2_base += np.sum((square_baseline < 0) & (square_baseline >= -0.5))
                            zer_base += np.sum(square_baseline == 0)
                            tot_base += square_baseline.shape[0] * square_baseline.shape[1]

                        if len(baseline_features) != 0:
                            # count baseline wave attr
                            pos1_base += np.sum(baseline_features > 0.5)
                            neg1_base += np.sum(baseline_features < -0.5)
                            pos2_base += np.sum((baseline_features > 0) & (baseline_features <= 0.5))
                            neg2_base += np.sum((baseline_features < 0) & (baseline_features >= -0.5))
                            zer_base += np.sum(baseline_features == 0)
                            tot_base += baseline_features.shape[0] * baseline_features.shape[1]

                    plot_dic['XAI'].extend([xai] * 15)
                    plot_dic['Wave'].extend(['Sine'] * 5)
                    plot_dic['Wave'].extend(['Base'] * 5)
                    plot_dic['Wave'].extend(['Square'] * 5)
                    plot_dic['Area'].extend(['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)'] * 3)
                    plot_dic['Percent'].extend([(pos1_sine / tot_sine) * 100,
                                                (pos2_sine / tot_sine) * 100,
                                                (zer_sine / tot_sine) * 100,
                                                (neg2_sine / tot_sine) * 100,
                                                (neg1_sine / tot_sine) * 100,
                                                (pos1_base / tot_base) * 100,
                                                (pos2_base / tot_base) * 100,
                                                (zer_base / tot_base) * 100,
                                                (neg2_base / tot_base) * 100,
                                                (neg1_base / tot_base) * 100,
                                                (pos1_squa / tot_squa) * 100,
                                                (pos2_squa / tot_squa) * 100,
                                                (zer_squa / tot_squa) * 100,
                                                (neg2_squa / tot_squa) * 100,
                                                (neg1_squa / tot_squa) * 100])
                    # ['#f5f3ef', '#e1f2ca', '#b6de87', '#80bb47', '#4f9125', '#376319']
                    df = pd.DataFrame(plot_dic)

                    order = ['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)']
                    chart = alt.Chart(df, title=alt.Title(xai, fontSize=20)).transform_calculate(
                        order=f"-indexof({order}, datum.Area)"
                    ).mark_bar().encode(
                        x='Wave:O',
                        y=alt.Y('sum(Percent):Q', scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color('Area:N', scale=alt.Scale(domain=order,
                                                                  range=['#376319', '#4f9125', '#f5f3ef', '#c0267e',
                                                                         '#8a1959'])),
                        order="order:Q"
                    ).properties(
                        width=50,
                        height=150
                    )

                    plots.append(chart)
                    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots')):
                        os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots'))

                    # chart.save(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots',
                    #                        xai + '_class_' + str(cl) + '_' + c + '_pos_zero_neg.png'),
                    #           engine="vl-convert", ppi=300)

                final_chart = ((plots[0] | plots[1] | plots[2] | plots[3] | plots[4]) & (
                        plots[5] | plots[6] | plots[7] | plots[8] | plots[9])) \
                    .configure_axis(labelFontSize=17, titleFontSize=20). \
                    configure_legend(titleFontSize=20, labelFontSize=17)

                final_chart.save(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots',
                                              'class_' + str(cl) + '_' + c + '_pos_zero_neg.png'),
                                 engine="vl-convert", ppi=300)


def create_boxplots_over_all_samples_and_iterations(additional_info_path, data_type, dataset_name, save_folder):
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)

    if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots')):
        os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots'))
    # iterate over classes
    for cl in ['1']:  # list(dic.keys()):
        if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl)):
            os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl))
        first = True
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        # create medians over all iterations
        for c in ['Correct']:  # list(sub_dic.keys()):
            if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c)):
                os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c))
            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                samples = list(set([os.path.split(os.path.split(os.path.split(i)[0])[0])[-1] for i in cl_list]))
                first = False

            first_sine = True
            first_baseline = True
            first_square = True
            plots = []
            for xai in ['IntegratedGradients', 'DeepLift', 'GuidedBackprop', 'Deconvolution', 'Occlusion',
                        'ShapleyValueSampling',
                        'KernelShap', 'LRP-Epsilon', 'LRP-Alpha1-Beta0', 'Lime']:  # xai_methods:
                plot_dic = {'Wave': [], 'Values': []}
                if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c, xai)):
                    os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c, xai))
                print(xai)
                xai_sublist = [s for s in cl_list if xai + '_' in s]
                for sample in samples:
                    sample_sublist = [s for s in xai_sublist if sample in s]
                    # iterate over iteration

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

                    # plot_dic['XAI'].extend(
                    #    [xai] * (len(sine_all_features) + len(baseline_all_features) + len(square_all_features)))
                    plot_dic['Wave'].extend(['Sine'] * len(sine_all_features))
                    plot_dic['Wave'].extend(['Base'] * len(baseline_all_features))
                    plot_dic['Wave'].extend(['Square'] * len(square_all_features))
                    plot_dic['Values'].extend([np.round(i, 1) for i in sine_all_features.tolist()])
                    plot_dic['Values'].extend([np.round(i, 1) for i in baseline_all_features.tolist()])
                    plot_dic['Values'].extend([np.round(i, 1) for i in square_all_features.tolist()])

                df = pd.DataFrame(plot_dic)
                chart = alt.Chart(df).mark_boxplot(ticks=True).encode(
                    x=alt.X("Wave:O", title=None, axis=alt.Axis(labels=False, ticks=False),
                            scale=alt.Scale(padding=1)),
                    y=alt.Y("Values:Q"),
                    color="Wave:N"
                    # column=alt.Column('XAI:N', header=alt.Header(orient='bottom'))
                ).properties(
                    width=100,
                    height=150,
                    title=xai
                )

                plots.append(chart)

                # chart.save(os.path.join(save_folder, data_type, dataset_name, 'xai_boxplots', cl, c, xai, 'total_boxplots.png'),
                #           ppi=300)

                # del chart

            final_chart = ((plots[0] | plots[1] | plots[2] | plots[3] | plots[4]) &
                           (plots[5] | plots[6] | plots[7] | plots[8] | plots[9])).configure_axis(
                labelFontSize=15, titleFontSize=15).configure_legend(
                titleFontSize=15, labelFontSize=15)

            final_chart.save(os.path.join(save_folder, data_type, dataset_name,
                                          'total_boxplots.png'))


def create_grouped_pos_neg_zero_stacked_bar_image_data(additional_info_path, data_type, dataset_name, save_folder):
    # load
    with open(os.path.join(save_folder, data_type, dataset_name, 'all_correct_wrong_rel_file_paths.pkl'), 'rb') as f:
        dic = pickle.load(f)
    first = True
    # iterate over classes
    for cl in ['1']:  # list(dic.keys()):
        # iterate over all, correct, wrong classified sample paths
        sub_dic = dic[cl]
        for c in ['Correct']:  # list(sub_dic.keys()):

            cl_list = sub_dic[c]
            # get list of xai methods
            if first:
                xai_methods = list(set([os.path.split(i)[-1].split('_')[0] for i in cl_list]))
                first = False
            if len(cl_list) != 0:
                plots = []
                # iterate over XAI methods
                for xai in ['LRP-Alpha1-Beta0', 'GuidedBackprop', 'IntegratedGradients', 'KernelShap', 'Occlusion',
                            'ShapleyValueSampling']:  # xai_methods:
                    counts = {
                        '1. Microaneurysms': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                              'total': 0},
                        '2. Haemorrhages': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                            'total': 0},
                        '3. Hard Exudates': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                             'total': 0},
                        '4. Soft Exudates': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                             'total': 0},
                        '5. Optic Disc': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                          'total': 0},
                        'Outside Mask': {'(0.5, 1]': 0, '(0, 0.5]': 0, '0': 0, '[-0.5, 0)': 0, '[-1, -0.5)': 0,
                                         'total': 0}}

                    print(xai)
                    plot_dic = {'XAI': [], 'Area': [], 'Type': [], 'Percent': []}
                    xai_sublist = [s for s in cl_list if xai + '_' in s]
                    # get information for each xai methods
                    for path in xai_sublist:
                        rel = pd.read_csv(path).to_numpy()
                        sample_with_grade = [sp.split('/') for sp in path.split('\\')][-3][0]
                        sample = sample_with_grade.split('_')[0] + '_' + sample_with_grade.split('_')[1]
                        # get segmentation masks
                        mas = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '1. Microaneurysms', sample + '_MA.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        mas[mas != 0] = 1
                        mas_mask = np.where((mas == 0) | (mas == 1), mas ^ 1, mas)
                        mas_values = ma.compressed(ma.masked_array(rel, mask=mas_mask))
                        counts['1. Microaneurysms']['(0.5, 1]'] += np.sum(mas_values > 0.5)
                        counts['1. Microaneurysms']['[-1, -0.5)'] += np.sum(mas_values < -0.5)
                        counts['1. Microaneurysms']['(0, 0.5]'] += np.sum((mas_values > 0) & (mas_values <= 0.5))
                        counts['1. Microaneurysms']['[-0.5, 0)'] += np.sum((mas_values < 0) & (mas_values >= -0.5))
                        counts['1. Microaneurysms']['0'] += np.sum(mas_values == 0)
                        counts['1. Microaneurysms']['total'] += len(mas_values)

                        exs = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '3. Hard Exudates', sample + '_EX.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        exs[exs != 0] = 1
                        exs_mask = np.where((exs == 0) | (exs == 1), exs ^ 1, exs)
                        exs_values = ma.compressed(ma.masked_array(rel, mask=exs_mask))
                        counts['3. Hard Exudates']['(0.5, 1]'] += np.sum(exs_values > 0.5)
                        counts['3. Hard Exudates']['[-1, -0.5)'] += np.sum(exs_values < -0.5)
                        counts['3. Hard Exudates']['(0, 0.5]'] += np.sum((exs_values > 0) & (exs_values <= 0.5))
                        counts['3. Hard Exudates']['[-0.5, 0)'] += np.sum((exs_values < 0) & (exs_values >= -0.5))
                        counts['3. Hard Exudates']['0'] += np.sum(exs_values == 0)
                        counts['3. Hard Exudates']['total'] += len(exs_values)

                        ods = cv2.imread(
                            os.path.join(additional_info_path, dataset_name, '5. Optic Disc', sample + '_OD.tif'),
                            cv2.IMREAD_GRAYSCALE)
                        ods[ods != 0] = 1
                        ods_mask = np.where((ods == 0) | (ods == 1), ods ^ 1, ods)
                        ods_values = ma.compressed(ma.masked_array(rel, mask=ods_mask))
                        counts['5. Optic Disc']['(0.5, 1]'] += np.sum(ods_values > 0.5)
                        counts['5. Optic Disc']['[-1, -0.5)'] += np.sum(ods_values < -0.5)
                        counts['5. Optic Disc']['(0, 0.5]'] += np.sum((ods_values > 0) & (ods_values <= 0.5))
                        counts['5. Optic Disc']['[-0.5, 0)'] += np.sum((ods_values < 0) & (ods_values >= -0.5))
                        counts['5. Optic Disc']['0'] += np.sum(ods_values == 0)
                        counts['5. Optic Disc']['total'] += len(ods_values)

                        if sample + '_HE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages')):
                            hes = cv2.imread(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages', sample + '_HE.tif'),
                                cv2.IMREAD_GRAYSCALE)
                            hes[hes != 0] = 1
                            hes_mask = np.where((hes == 0) | (hes == 1), hes ^ 1, hes)
                            hes_values = ma.compressed(ma.masked_array(rel, mask=hes_mask))

                            counts['2. Haemorrhages']['(0.5, 1]'] += np.sum(hes_values > 0.5)
                            counts['2. Haemorrhages']['[-1, -0.5)'] += np.sum(hes_values < -0.5)
                            counts['2. Haemorrhages']['(0, 0.5]'] += np.sum((hes_values > 0) & (hes_values <= 0.5))
                            counts['2. Haemorrhages']['[-0.5, 0)'] += np.sum((hes_values < 0) & (hes_values >= -0.5))
                            counts['2. Haemorrhages']['0'] += np.sum(hes_values == 0)
                            counts['2. Haemorrhages']['total'] += len(hes_values)

                        if sample + '_SE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                            ses = cv2.imread(
                                os.path.join(additional_info_path, dataset_name, '4. Soft Exudates',
                                             sample + '_SE.tif'),
                                cv2.IMREAD_GRAYSCALE)
                            ses[ses != 0] = 1
                            ses_mask = np.where((ses == 0) | (ses == 1), ses ^ 1, ses)
                            ses_values = ma.compressed(ma.masked_array(rel, mask=ses_mask))
                            counts['4. Soft Exudates']['(0.5, 1]'] += np.sum(ses_values > 0.5)
                            counts['4. Soft Exudates']['[-1, -0.5)'] += np.sum(ses_values < -0.5)
                            counts['4. Soft Exudates']['(0, 0.5]'] += np.sum((ses_values > 0) & (ses_values <= 0.5))
                            counts['4. Soft Exudates']['[-0.5, 0)'] += np.sum((ses_values < 0) & (ses_values >= -0.5))
                            counts['4. Soft Exudates']['0'] += np.sum(ses_values == 0)
                            counts['4. Soft Exudates']['total'] += len(ses_values)

                        if sample + '_HE.tif' in os.listdir(
                                os.path.join(additional_info_path, dataset_name, '2. Haemorrhages')):
                            if sample + '_SE.tif' in os.listdir(
                                    os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                                normal_mask = hes + exs + mas + ods + ses
                            else:
                                normal_mask = hes + exs + mas + ods
                        else:
                            if sample + '_SE.tif' in os.listdir(
                                    os.path.join(additional_info_path, dataset_name, '4. Soft Exudates')):
                                normal_mask = exs + mas + ods + ses
                            else:
                                normal_mask = exs + mas + ods
                        normal_mask[normal_mask != 0] = 1
                        normal_values = ma.compressed(ma.masked_array(rel, mask=normal_mask))
                        counts['Outside Mask']['(0.5, 1]'] += np.sum(normal_values > 0.5)
                        counts['Outside Mask']['[-1, -0.5)'] += np.sum(normal_values < -0.5)
                        counts['Outside Mask']['(0, 0.5]'] += np.sum((normal_values > 0) & (normal_values <= 0.5))
                        counts['Outside Mask']['[-0.5, 0)'] += np.sum((normal_values < 0) & (normal_values >= -0.5))
                        counts['Outside Mask']['0'] += np.sum(normal_values == 0)
                        counts['Outside Mask']['total'] += len(normal_values)

                    plot_dic['XAI'].extend([xai] * 30)
                    plot_dic['Type'].extend(
                        ['MA'] * 5 + ['HE'] * 5 + ['EX'] * 5 + ['SE'] * 5 + ['OD'] * 5 + ['Normal'] * 5)
                    plot_dic['Area'].extend(['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)'] * 6)

                    for key in counts:
                        areas = list(counts[key].keys())[:5]
                        for area in areas:
                            plot_dic['Percent'].append((counts[key][area] / counts[key]['total']) * 100)

                    # ['#f5f3ef', '#e1f2ca', '#b6de87', '#80bb47', '#4f9125', '#376319']
                    df = pd.DataFrame(plot_dic)

                    order = ['(0.5, 1]', '(0, 0.5]', '0', '[-0.5, 0)', '[-1, -0.5)']
                    chart = alt.Chart(df, title=alt.Title(xai, fontSize=20)).transform_calculate(
                        order=f"-indexof({order}, datum.Area)"
                    ).mark_bar().encode(
                        x='Type:O',
                        y=alt.Y('sum(Percent):Q', scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color('Area:N', scale=alt.Scale(domain=order,
                                                                  range=['#376319', '#4f9125', '#f5f3ef', '#c0267e',
                                                                         '#8a1959'])),
                        order="order:Q"
                    ).properties(
                        width=100,
                        height=150
                    )

                    plots.append(chart)
                    # if not os.path.exists(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots')):
                    #    os.makedirs(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots'))

                    # chart.save(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots',
                    #                        xai + '_class_' + str(cl) + '_' + c + '_pos_zero_neg.png'),
                    #           engine="vl-convert", ppi=300)

                final_chart = ((plots[0] | plots[1] | plots[2]) & (
                        plots[3] | plots[4] | plots[5])) \
                    .configure_axis(labelFontSize=17, titleFontSize=20). \
                    configure_legend(titleFontSize=20, labelFontSize=17)

                final_chart.save(os.path.join(save_folder, data_type, dataset_name, 'pos_zero_neg_plots',
                                              'class_' + str(cl) + '_' + c + '_pos_zero_neg.png'),
                                 engine="vl-convert", ppi=300)



#create_rank_stacked_bar(root, 'max', 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_wrong=True)
#create_rank_stacked_bar(root, 'max', 'tabular_data', 'heart_failure_clinical_records', save, correct_wrong=True)
#create_rank_stacked_bar(root, 'max', 'tabular_data', 'tcga_brca', save, correct_wrong=True)

#create_rank_stacked_bar(root, 'min', 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_wrong=True)
#create_rank_stacked_bar(root, 'min', 'tabular_data', 'heart_failure_clinical_records', save, correct_wrong=True)
#create_rank_stacked_bar(root, 'min', 'tabular_data', 'tcga_brca', save, correct_wrong=True)

#create_wilcoxon_sums_heatmap(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
#create_wilcoxon_sums_heatmap(root, 'tabular_data', 'heart_failure_clinical_records', save)

#create_rf_logreg_boxplot(root, 'tabular_data', 'breast_cancer_wisconsin_data', save)
#create_rf_logreg_boxplot(root, 'tabular_data', 'heart_failure_clinical_records', save)
#create_rf_logreg_boxplot(root, 'tabular_data', 'tcga_brca', save)

# create_rank_stacked_bar(root, 'max', 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_wrong=True)
# create_rank_stacked_bar(root, 'max', 'tabular_data', 'heart_failure_clinical_records', save, correct_wrong=True)
# create_rank_stacked_bar(root, 'max', 'tabular_data', 'tcga_brca', save, correct_wrong=True)

# create_rank_stacked_bar(root, 'min', 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_wrong=True)
# create_rank_stacked_bar(root, 'min', 'tabular_data', 'heart_failure_clinical_records', save, correct_wrong=True)
# create_rank_stacked_bar(root, 'min', 'tabular_data', 'tcga_brca', save, correct_wrong=True)

#create_pos_neg_zero_stacked_bar(root, 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_classified_only=True, wrong_classified_only=False)
#create_pos_neg_zero_stacked_bar(root, 'tabular_data', 'heart_failure_clinical_records', save, correct_classified_only=True, wrong_classified_only=False)

#create_stacked_density_plot(root, 'tabular_data', 'breast_cancer_wisconsin_data', save, correct_classified_only=True, wrong_classified_only=False)
#create_stacked_density_plot(root, 'tabular_data', 'heart_failure_clinical_records', save, correct_classified_only=True, wrong_classified_only=False)

# create_sig_rel_plots(root, data, 'signal_data', 'synthetic_biosignal_data', save, norm=False)
# create_median_sig_rel_plots(data, 'signal_data', 'synthetic_biosignal_data', save)
create_grouped_pos_neg_zero_stacked_bar_signal_data(add_info, 'signal_data', 'synthetic_biosignal_data', save)
# create_median_sample_wise_boxplots(add_info, 'signal_data', 'synthetic_biosignal_data', save)
# create_boxplots_over_all_samples_and_iterations(add_info, 'signal_data', 'synthetic_biosignal_data', save)
#create_median_sig_rel_plot_of_only_sine_and_square_waves('class_1_sample_4093', add_info, data, 'signal_data', 'synthetic_biosignal_data', save)
# plot_example_signal('class_1_sample_4093', data, 'signal_data', 'synthetic_biosignal_data', save)
# create_normalized_relevance_heatmap_from_medians_csv("E:/Uni/Doktor-Goettingen/Datasets/benchmark/results_plots/tabular_data/tcga_brca/class_1/median_class_1_pam50_correctly_classified.csv", 'tabular_data', 'tcga_brca', save)
#create_img_rel_plots(data, 'image_data', 'retina', 'IDRiD_223_grade_4', save)
# create_grouped_pos_neg_zero_stacked_bar_image_data(add_info, 'image_data', 'retina', save)
# create_boxplots_over_all_samples_and_iterations_image_data(add_info, 'image_data', 'retina', save)
