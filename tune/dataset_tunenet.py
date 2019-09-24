#!/usr/bin/env python
#

#
# Generate a "tuning" datset, where each datapoint in the set consists of the information from two bouncing ball
# simulators. Used to train TuneNet.

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from tune.utils import get_torch_device, get_dataset_base_path, get_immediate_subdirectories

device = get_torch_device()


class DatasetTuneNet(Dataset):
    dataset_max = torch.tensor([5.0, 100.0, 5.0], device=device, dtype=torch.float64).unsqueeze(1).repeat(1, 400)
    dataset_min = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64).unsqueeze(1).repeat(1, 400)

    # dataset of videos, positions, and physics parameters for dropped balls.

    def __init__(self, dataset_name, observation_type, transform=None):
        """
        :param dataset_name: directory containing directories 0, 1, 2, etc. (one folder per simulation)
        :param transform: Optional transform(s) to be applied on a sample
        """
        self.root_dir = get_dataset_base_path(dataset_name)
        print("this dataset exists in " + self.root_dir)
        self.loadtype = observation_type
        self.transform = transform

        self.length = len(get_immediate_subdirectories(self.root_dir))
        print('dataset loaded from {} with {} elements'.format(self.root_dir, self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load ground truth
        def make_name(name):
            return osp.join(self.root_dir, str(idx), name + ".npy")

        if self.loadtype == "ground_truth":
            zeta1 = (np.load(make_name("physics1"))[0], np.load(make_name("start_position"))[0])
            zeta2 = (np.load(make_name("physics2"))[0], np.load(make_name("start_position"))[0])
            zeta = torch.tensor(np.vstack((zeta1, zeta2))).to(device)

            s1 = np.load(make_name("position1"))[:, 2]
            s2 = np.load(make_name("position2"))[:, 2]
            s = torch.tensor(np.vstack((s1, s2))).to(device)

        elif self.loadtype == "observation":
            zeta1 = (np.load(make_name("physics1"))[0], np.load(make_name("start_position"))[0])
            zeta2 = (np.load(make_name("physics2"))[0], np.load(make_name("start_position"))[0])
            zeta = torch.tensor(np.vstack((zeta1, zeta2))).to(device)

            s1 = np.load(make_name("position1"))[:, 2]
            s2 = np.load(make_name("center2"))[:, 1]
            s3 = np.load(make_name("position2"))[:, 2]
            s = torch.tensor(np.vstack((s1, s2, s3))).to(device)
        else:
            print("Loadtype {} not understood.".format(self.loadtype))

        if self.transform:
            s[:, :] = self.transform(s[:, :])

        v1 = np.load(make_name("linear_velocity_list1"))[:, 2]
        v2 = np.load(make_name("linear_velocity_list2"))[:, 2]
        v = torch.tensor(np.vstack((v1, v2))).to(device)

        return [zeta, s, v]

    @classmethod
    def get_data_loader(cls, dataset_name: str, observation_type: str, dataset_action: str, batch_size: int):
        """
        Get a fully-fledged DataLoader object that can be used to iterate over a given dataset.
        :param dataset_name: The overall name of the dataset to load. The dataset exists in a dir with this name.
        :param observation_type: A string, either "ground_truth" or "observation," to help the network apply
                                 transformations to the input and convert it to the range [-1, 1].
        :param dataset_action: train, test, or val. The actual datapoints will be stored in a subdir with this name
                               inside the overall dataset folder
        :param batch_size: Number of datapoints to load into each batch
        :return:
        """
        transform = transforms.Compose([])
        if observation_type == "observation":
            transform = transforms.Compose([
                transforms.Lambda(lambda x: (x - cls.dataset_min) / cls.dataset_max)
            ])
        dataset = DatasetTuneNet(dataset_name=dataset_name + "/" + dataset_action,
                                 transform=transform,
                                 observation_type=observation_type)

        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size)


class GaussianNoise(object):
    """Rescale the image in a sample to a given size.

    Args:
        stdev (float): standard deviation of gaussian noise.
    """

    def __init__(self, stdev=0.1):
        self.stdev = stdev
        self.noise = torch.tensor(0, dtype=torch.double).to(device)

    def __call__(self, x):
        sampled_noise = self.noise.repeat(*x.size()).normal_(self.stdev)
        return x + sampled_noise


class DatasetTuneNetKinova(Dataset):
    # dataset_max = torch.tensor([5.0, 100.0, 5.0], device=device, dtype=torch.float64).unsqueeze(1).repeat(1, 400)
    # dataset_min = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64).unsqueeze(1).repeat(1, 400)
    # dataset of videos, positions, and physics parameters for dropped balls.

    def __init__(self, dataset_name, transform=None):
        """
        :param root_dir: directory containing directories 0, 1, 2, etc. (one folder per simulation)
        :param transform: Optional transform(s) to be applied on a sample
        """
        self.root_dir = get_dataset_base_path(dataset_name)
        print("this dataset exists in " + self.root_dir)
        self.transform = transform

        self.length = len(get_immediate_subdirectories(self.root_dir))
        print('dataset loaded from {} with {} elements'.format(self.root_dir, self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # load ground truth
        def make_name(name):
            return osp.join(self.root_dir, str(idx), name + ".npy")

        zeta1 = np.load(make_name("mass1"))
        zeta2 = np.load(make_name("mass2"))
        zeta = torch.tensor(np.stack((zeta1, zeta2), axis=0)).to(device)

        o1 = np.load(make_name("torques1"))
        o2 = np.load(make_name("torques2"))
        o = torch.tensor(np.stack((o1, o2), axis=0)).to(device)

        if self.transform:
            o = self.transform(o)

        return [zeta, o]

    @classmethod
    def load_dataset_limits(cls, path):
        limits_filename = os.path.join(path, "limits.npy")
        # get data size
        subdirs = get_immediate_subdirectories(path)
        single_torque_size = np.load(os.path.join(path, subdirs[0], "torques1.npy")).shape
        print(single_torque_size)
        if os.path.isfile(limits_filename):
            loaded = np.load(limits_filename)
            print(loaded)
            mean = loaded["mean"]
            std = loaded["std"]
            print("loaded dataset mean: {}, std: {}".format(mean, std))
            mean_tensor = torch.tensor(mean).unsqueeze(0).repeat(single_torque_size[0]*2, 1).to(device)
            std_tensor = torch.tensor(std).unsqueeze(0).repeat(single_torque_size[0]*2, 1).to(device)
            # print("mean tensor shape")
            # print(mean_tensor.shape)
            return mean_tensor, std_tensor
        else:
            # we need to calculate the limits
            torques = np.empty([len(subdirs)*2, *single_torque_size])

            idx = 0
            for subdir in subdirs:
                for torque_file in "torques1.npy", "torques2.npy":
                    torques_loaded = np.load(os.path.join(path, subdir, torque_file))
                    torques[idx] = torques_loaded
                    idx += 1

            mean = np.mean(torques, axis=(0, 1), keepdims=False)
            std = np.std(torques, axis=(0, 1), keepdims=False)
            # print("calculated dataset mean: {}, std: {}".format(mean, std))
            # these gnarly lines spread the mean and std tensors so they are the shape of the loaded data structure
            mean_tensor = torch.tensor(mean).unsqueeze(0).repeat(single_torque_size[0], 1).unsqueeze(0).repeat(2, 1, 1).to(device)
            std_tensor = torch.tensor(std).unsqueeze(0).repeat(single_torque_size[0], 1).unsqueeze(0).repeat(2, 1, 1).to(device)
            # print("mean tensor shape")
            # print(mean_tensor.shape)
            np.savez(limits_filename, allow_pickle=False, mean=mean, std=std)
            return mean_tensor, std_tensor

    @classmethod
    def get_data_loader(cls, dataset_name: str, dataset_action: str, batch_size: int):
        """
        Get a fully-fledged DataLoader object that can be used to iterate over a given dataset.
        :param dataset_name: The overall name of the dataset to load. The dataset exists in a dir with this name.
        :param dataset_action: train, test, or val. The actual datapoints will be stored in a subdir with this name
                               inside the overall dataset folder
        :param batch_size: Number of datapoints to load into each batch
        :return:
        """
        dataset_dir = get_dataset_base_path(dataset_name)
        dataset_mean, dataset_std = cls.load_dataset_limits(
            os.path.join(dataset_dir, "train"))
        transform = transforms.Compose([
            transforms.Lambda(lambda x: (x - dataset_mean) / dataset_std)
        ])
        dataset = DatasetTuneNetKinova(dataset_name=dataset_name + "/" + dataset_action,
                                 transform=transform)
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=batch_size)