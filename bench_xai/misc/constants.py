"""
    Constant values used in the code.
    AVAILABLE_XAI_MODELS
    PERFORMANCE_METRICS

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinarySpecificity, \
    BinaryF1Score, BinaryMatthewsCorrCoef, BinaryAveragePrecision

AVAILABLE_XAI_MODELS = [
        'IntegratedGradients',  # default 0 baseline
        'DeepLift',  # default 0 baseline, eps=1e-10, multiply by inputs=True
        'DeepLiftShap',
        'GradientShap',
        'InputXGradient',
        'GuidedBackprop',
        'Deconvolution',
        'Occlusion',
        'ShapleyValueSampling',
        'KernelShap',
        'LRP',
        'LRP-Epsilon',
        'LRP-Alpha1-Beta0',
        'LRP-Gamma',
        'Lime'
    ]

# evaluation metrics that will get calculated during training and validation of AI model
PERFORMANCE_METRICS = [
        'Accuracy',
        'AUROC',
        'Sensitivity',
        'Specificity',
        'F1',
        'MCC',
        'AUPRC'
]
