"""
    Tabular dataset class.
    Allows loading of csv files into Dataset type.
    Columns need to be features and rows samples.
    The last column needs to be the target feature.

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):

    def __init__(self, file_path, target_name=None, sep=','):  # constructor
        # check that file ends in .csv
        assert file_path.split('.')[len(file_path.split('.'))-1] == 'csv', \
            f"CSV file expected, got {file_path.split('.')[len(file_path.split('.'))-1]}"

        # read in file
        df = pd.read_csv(file_path, sep=sep)

        # check that target feature name is in columns list
        assert target_name in df.columns, \
            f"Expected target_name to be in df.columns. Got {target_name}."

        # move target feature to last column
        df[target_name] = df.pop(target_name)
        # get features (excluding target feature which is in the last column)
        x = df.iloc[:, :-1].values
        # get target feature
        y = df.iloc[:, -1:].values
        # get feature names
        cols = df.iloc[:, :-1].columns.tolist()
         
        self.data = torch.tensor(x, dtype=torch.float32)
        self.label = torch.tensor(y, dtype=torch.float32).squeeze(-1).type(torch.LongTensor)
        self.feat_names = cols
        # create index for all samples (from 0 to number of samples)
        self.file_names = ['Sample_'+str(i) for i in range(len(self.label))]

    def get_features(self):
        return self.feat_names

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.file_names[idx]
