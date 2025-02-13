"""
    Example code for monte carlo cross validation.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2022 HauschildLab group
    :date: 2022-09-15
"""

import os
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, ones_
import bench_xai
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# implement neural network
class BreastCancerMLP(nn.Module):

    def __init__(self, n_features):
        super(BreastCancerMLP, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(n_features, 20)
        self.dropout1 = nn.Dropout(0.2)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(10, 2)

        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        xavier_uniform_(self.fc3.weight)

        ones_(self.fc1.bias)
        ones_(self.fc2.bias)
        ones_(self.fc3.bias)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

# Set correct root_path, data_type and dataset_name -------------------------------------------------------------------------
root = "E:/benchmark"
data_type = "tabular_data"
dataset_name = "breast_cancer_wisconsin_data"

# get the path where your data file is saved
# (for us it is "E:/benchmark/data/tabular_data/breast_cancer_wisconsin_data/file.csv")
file_path = os.path.join(os.path.join(root, 'data', data_type, dataset_name),
                         os.listdir(os.path.join(root, 'data', data_type, dataset_name))[0])

# load the dataset into memory -------------------------------------------------------------------
dataset = bench_xai.TabularDataset(file_path=file_path, target_name='Target', sep=',')


# load available models for this dataset
nn_model = BreastCancerMLP(len(dataset.get_features()))
loss = nn.CrossEntropyLoss()
epochs = 300
batchsize = 128
opti = torch.optim.Adam(nn_model.parameters(), lr=0.001)

log_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(random_state=42)


# start training neural network ------------------------------------------------------------------
bench_xai.mccv_loop(result_path=os.path.join(root, 'results'), data=dataset, dataset_name=dataset_name,
                    data_type=data_type, model=nn_model, model_type="pytorch", loss_func=loss, epochs=epochs,
                    batchsize=batchsize, opti=opti, mccv_start_iteration=0, mccv_end_iteration=100,
                    validation_split=0.2, test_samples=50, random_seed=42,
                    xai_methods=bench_xai.AVAILABLE_XAI_MODELS, train=True, config_path=os.path.join(root, "example_xai_config.cfg"), device=device)

# start training logistic regression -------------------------------------------------------------
bench_xai.mccv_loop(result_path=os.path.join(root, 'results'), data=dataset, dataset_name=dataset_name,
                    data_type=data_type, model=log_model, model_type="logreg", mccv_start_iteration=0,
                    mccv_end_iteration=100, validation_split=0.2, test_samples=50,
                    random_seed=42, xai_methods=bench_xai.AVAILABLE_XAI_MODELS, train=True, device=device)

# start training rf -------------------------------------------------------------
bench_xai.mccv_loop(result_path=os.path.join(root, 'results'), data=dataset, dataset_name=dataset_name,
                    data_type=data_type, model=rf_model, model_type="rf", mccv_start_iteration=0,
                    mccv_end_iteration=100, validation_split=0.2, test_samples=50,
                    random_seed=42, xai_methods=bench_xai.AVAILABLE_XAI_MODELS, train=True, device=device)

