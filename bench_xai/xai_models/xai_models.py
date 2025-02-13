""""
    Function that creates Feature Attributions for selected XAI model.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import os
import time
import configparser
import torch
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, GradientShap, InputXGradient, \
    GuidedBackprop, Deconvolution, FeatureAblation, Occlusion, ShapleyValueSampling, \
    KernelShap, LRP, Lime, LimeBase

from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from bench_xai.misc.helper_functions import set_rule_attr, create_super_pixel_mask


def get_rel(xai_model,  model, data_type, inputs, target_index,
            config_path=None, device='cuda'):
    """
    Function that returns attributions for specified xai model and inputs.

    :param xai_model: XAI model to use for getting attributions
    :param model: Trained neural network
    :param data_type: data type (either 'image_data','tabular_data', 'signal_data')
    :param inputs: Inputs to create attributions for
    :param target_index: list of target indexes for which relevance is computed (for class 0 index should be zero, for
                         class 1 index=1, etc)
    :param config_path: path to a config file (default: None), None uses default parameters for XAI methods
    :param device: where to put variables ('cuda' if available or 'cpu')

    :return: Attributions in the form of the inputs and runtime of XAI method in seconds
    """
    # load config file if path is given
    if config_path:
        # load config file for xai methods
        config = configparser.ConfigParser()
        config.read(config_path)

    # init gradients
    inputs.requires_grad_()

    if xai_model == 'IntegratedGradients':
        # init xai model
        xai = IntegratedGradients(model)
        # set default baseline (0)
        baseline = 0
        # get scalar baseline from config file
        if config_path:
            baseline = int(config.get('IntegratedGradients', 'baseline'))

        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'DeepLift':
        # init xai model
        xai = DeepLift(model)
        # set default baseline (0)
        baseline = 0
        # get scalar baseline from config file
        if config_path:
            baseline = int(config.get('DeepLift', 'baseline'))

        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'DeepLiftShap':
        # init xai model
        xai = DeepLiftShap(model)
        # set default baseline (0)
        baseline = torch.full(inputs.shape, float(0)).to(device)
        # get scalar baseline from config file
        if config_path:
            baseline = torch.full(inputs.shape, float(config.get('DeepLiftShap', 'baseline'))).to(device)

        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'GradientShap':
        # init xai model
        xai = GradientShap(model)
        # set default baseline (0)
        baseline = torch.full(inputs.shape, float(0)).to(device)
        # get scalar baseline from config file
        if config_path:
            baseline = torch.full(inputs.shape, float(config.get('GradientShap', 'baseline'))).to(device)

        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'InputXGradient':
        # init xai model
        xai = InputXGradient(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'GuidedBackprop':
        # init xai model
        xai = GuidedBackprop(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'Deconvolution':
        # init xai model
        xai = Deconvolution(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'Occlusion':
        if data_type in ['image_data', 'signal_data']:
            # init xai model
            xai = Occlusion(model)
            # get stride step size
            strides = 3
            # get scalar baseline from config file
            baseline = 0
            if config_path:
                # get stride step size
                strides = eval(config.get('Occlusion', 'strides'))
                # get scalar baseline from config file
                baseline = int(config.get('Occlusion', 'baseline'))

            # start attribution
            # check if image is 3D (batch_size, color, x_dim, y_dim) or 2D (batch_size, x_dim, y_dim)
            if len(inputs.shape) == 4:
                window = (3, 3, 3)
                if config_path:
                    window = eval(config.get('Occlusion', '3d_sliding_window_shapes'))

                # start timer
                start_time = time.time()

                attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index, strides=strides,
                                     sliding_window_shapes=window)
            else:
                window = (3, 3)
                if config_path:
                    window = eval(config.get('Occlusion', '2d_sliding_window_shapes'))
                # start timer
                start_time = time.time()

                attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index, strides=strides,
                                     sliding_window_shapes=window)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

        elif data_type == 'tabular_data':
            # init xai model
            xai = FeatureAblation(model)  # same as occlusion but for tabular data
            baseline = 0
            # get scalar baseline from config file
            if config_path:
                baseline = int(config.get('FeatureAblation', 'baseline'))
            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

    elif xai_model == 'ShapleyValueSampling':
        baseline = 0
        superpixelshape = (1, 4)
        n_samples = 25
        # get scalar baseline from config file
        if config_path:
            baseline = int(config.get('ShapleyValueSampling', 'baseline'))
            # get super pixel shape for image (height, width)
            superpixelshape = eval(config.get('ShapleyValueSampling', 'feature_mask'))
            # get number of feature permutations tested (default=25)
            n_samples = int(config.get('ShapleyValueSampling', 'n_samples'))
        # init xai model
        xai = ShapleyValueSampling(model)

        if data_type in ['image_data', 'signal_data']:
            # get superpixelmask
            super_pixel_mask = create_super_pixel_mask(inputs.shape, superpixelshape, device)

            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index,
                                 feature_mask=super_pixel_mask, n_samples=n_samples)
            # aggregate color channel by mean (attributions for each color channel are the same because of super pixels)
            # for the other xai methods the sum over color channels is calculated
            if len(attr.shape) == 4:
                attr = torch.mean(attr, dim=1)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

        elif data_type == 'tabular_data':
            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index, n_samples=n_samples)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

    elif xai_model == 'KernelShap':
        baseline = 0
        superpixelshape = (1, 4)
        n_samples = 50
        # get scalar baseline from config file
        if config_path:
            baseline = int(config.get('KernelShap', 'baseline'))
            # get super pixel shape for image (height, width)
            superpixelshape = eval(config.get('KernelShap', 'feature_mask'))
            # get number of feature permutations tested (default=25)
            n_samples = int(config.get('KernelShap', 'n_samples'))

        # init xai model
        xai = KernelShap(model)
        if data_type in ['image_data', 'signal_data']:
            # get superpixelmask
            super_pixel_mask = create_super_pixel_mask(inputs.shape, superpixelshape, device)

            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index,
                                 feature_mask=super_pixel_mask, n_samples=n_samples)
            # aggregate color channel by mean (attributions for each color channel are the same because of super pixels)
            # for the other xai methods the sum over color channels is calculated
            if len(attr.shape) == 4:
                attr = torch.mean(attr, dim=1)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

        elif data_type == 'tabular_data':
            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), baselines=baseline, target=target_index,  n_samples=n_samples)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

    elif xai_model == 'LRP':  # default epsilon rule for most layer types with e=1e-9
        for layer in model.children():
            set_rule_attr(layer, EpsilonRule()) # epsilon rule is default
        # init xai model
        xai = LRP(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'LRP-Epsilon':
        epsilon = 1e-4
        # get scalar epsilon from config file
        if config_path:
            epsilon = float(config.get('LRP-Epsilon', 'epsilon'))

        for layer in model.children():
            set_rule_attr(layer, EpsilonRule(epsilon=epsilon))

        # init xai model
        xai = LRP(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'LRP-Alpha1-Beta0':
        for layer in model.children():
            set_rule_attr(layer, Alpha1_Beta0_Rule())

        # init xai model
        xai = LRP(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'LRP-Gamma':
        gamma = 0.25
        # get scalar epsilon from config file
        if config_path:
            gamma = float(config.get('LRP-Gamma', 'gamma'))
        for layer in model.children():
            set_rule_attr(layer, GammaRule(gamma=gamma))

        # init xai model
        xai = LRP(model)
        # start timer
        start_time = time.time()
        # start attribution
        attr = xai.attribute(inputs.to(device), target=target_index)
        # calculate time passed in seconds
        time_sec = time.time() - start_time

    elif xai_model == 'Lime':
        superpixelshape = (1, 4)
        n_samples = 50
        if config_path:
            # get super pixel shape for image (height, width)
            superpixelshape = eval(config.get('Lime', 'feature_mask'))
            # get number of samples used to train surrogate model (default=50)
            n_samples = int(config.get('Lime', 'n_samples'))
        # init xai model
        xai = Lime(model)
        if data_type in ['image_data', 'signal_data']:
            # get superpixelmask
            super_pixel_mask = create_super_pixel_mask(inputs.shape, superpixelshape, device)
            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), target=target_index,
                                 feature_mask=super_pixel_mask, n_samples=n_samples)
            # aggregate color channel by mean (attributions for each color channel are the same because of super pixels)
            # for the other xai methods the sum over color channels is calculated
            if len(attr.shape) == 4:
                attr = torch.mean(attr, dim=1)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

        elif data_type == 'tabular_data':
            # start timer
            start_time = time.time()
            # start attribution
            attr = xai.attribute(inputs.to(device), target=target_index, n_samples=n_samples)
            # calculate time passed in seconds
            time_sec = time.time() - start_time

    return attr, time_sec
