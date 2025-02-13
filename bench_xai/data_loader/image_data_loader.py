"""
    Image dataset class.
    The dataset should have a root folder (name of the dataset)
    and inside a folder for each class containing images.
    All filenames should be unique, even across folders.
    Example folder structure:

    - cat_dog_dataset
        - cat
            - cat01.jpg
            - cat02.jpg
            ...
        - dog
            - dog01.jpg
            - dog02.jpg
            ...

    :author: Jacqueline Michelle Beinecke
    :copyright: © 2023 HauschildLab group
    :date: 2023-10-10
"""

import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, root_path, transform=None):  # constructor
        self.root_dir = root_path  # path to dataset folder
        self.transform = transform  # transformer for transforming images

        # create label tensor
        count = 0
        # init list to save file-paths
        f_paths = []
        # create label for every class_folder and save image names in tensor as well
        for dic in sorted(os.listdir(root_path)):
            print('Folder ', dic, ' turned to label ', count)
            files = [os.path.join(root_path, dic, file) for file in os.listdir(os.path.join(root_path, dic))]
            names = [file for file in os.listdir(os.path.join(root_path, dic))]

            # append image
            f_paths += files
            # create torch tensor with labels (size = amount images in that folder)
            lab = torch.full([len(files)], count)
            # only append new labels if there are more than 1
            if count > 0:
                label = torch.cat((label, lab), 0)
                file_names.extend(names)
            else:
                label = lab
                file_names = names
            # increase label counter
            count += 1

        # check that all file_names are unique
        assert len(file_names) == len(set(file_names)), "Not all image file names are unique! Make sure filenames " \
                                                        "are unique also across classes!"
        # reshape label list into array of dim (nr.of.samples,1)
        l = np.array(label).reshape(len(label), 1)

        self.label = torch.tensor(l, dtype=torch.float32).squeeze(-1).type(torch.LongTensor)
        self.images = f_paths

        self.file_names = file_names

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.images[idx].split('.')[-1] == 'npy':
            image = np.load(self.images[idx])
        else:
            # always load image as colored (3 channels) (x_dim, y_dim, color)
            image = cv2.imread(self.images[idx])
            # put color channel first
            image = np.moveaxis(image, -1, 0)

        if self.transform:
            image = self.transform(image)
        else:
            # image needs to be at least transformed to tensor
            image = torch.from_numpy(image).float()

        return image, self.label[idx], self.file_names[idx]  # the loaded filenames batch will be of type tuple

