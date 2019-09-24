#!/usr/bin/env python
#

#
# Useful functions for TuneNet.
import datetime
import errno
import os
from pathlib import PurePath

import numpy as np

import matplotlib.pyplot as plt
import pybullet as p
import torch
from tensorboardX import SummaryWriter
import joblib as joblib

from .definitions import ROOT_DIR, OUTPUT_DIR, DATASET_DIR


def get_torch_device():
    """
    Return the appropriate device for Torch: GPU if available with CUDA, otherwise the CPU
    :return: the device selected.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using Torch device: ' + str(device))
    return device


def get_dataset_base_path(dataset_name):
    """
    Return the base path for a dataset (which may or may not exist) given the dataset name.
    :param dataset_name: the name of the dataset
    :return: the absolute path to the dataset root
    """
    base_path = os.path.join(ROOT_DIR, DATASET_DIR, dataset_name)
    return base_path


def setup_physics(path=None, gui=False, verbose=False):
    """
    Also attempts to load the FileIOPlugin, which is required to use multiple search paths i.e.
    for URDFs. If successful, it adds the paths to the physics engine. The "objects" folder is automatically added.

    :param paths: a list of absolute and/or relative paths to add to the physics engine's search path.
                  if the paths are relative, they are added relative to the project root.
    :return: None
    """
    # handle args
    if path is None:
        path = "objects"

    if PurePath(path).is_absolute():
        abs_path = path
    else:
        abs_path = os.path.join(ROOT_DIR, path)

    # connect physics
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    # set path
    if verbose:
        print("setting search path to " + abs_path)
    p.setAdditionalSearchPath(abs_path)

    # add paths
    # file_io = p.loadPlugin("fileIOPlugin")
    # if file_io < 0:
    #     print("could not load FileIO plugin")
    # for path in paths:
    #     if PurePath(path).is_absolute():
    #         abs_path = path
    #     else:
    #         abs_path = os.path.join(ROOT_DIR, path)
    #     print("adding search path " + abs_path)
        # for root, directories, filenames in os.walk(abs_path):
        #     # for directory in directories:
        #     # 	print os.path.join(root, directory)
        #     for filename in filenames:
        #         abs_filename = os.path.join(root, filename)
        #         print("found file " + abs_filename)
        #         p.executePluginCommand(file_io, abs_filename, [p.AddFileIOAction, p.PosixFileIO])
    return p


def makedirs(directory):
    """
    Improved version of makedirs that will make a directory if it doesn't exist.
    :param directory: directory path to create
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        # don't get mad if the folder already exists.
        if e.errno != errno.EEXIST:
            raise


def setup_plots():
    """
    Basic plot setups for convenience and consistency
    :return:
    """
    plt.rcParams["font.family"] = "serif"


def make_video_path(task_path, itr):
    """
    Generate a video name based on a task name and iteration number
    :param task_path: the name of the task
    :param itr: the iteration number
    """
    return "{}_tune_iter{}.mp4".format(task_path.replace("_center", ""), itr)


def make_cor_path(task_path):
    return "{}_cor.npy".format(task_path.replace("_center", ""))


def save_model(model, epoch, prefix):
    """
    Save a set of PyTorch model weights associated with a training run.

    :param model: The model to grab weights from and save
    :param epoch: The training epoch (used in the output filename)
    :param prefix: The run prefix
    :return:
    """
    dirpath = os.path.join(ROOT_DIR, OUTPUT_DIR)
    makedirs(dirpath)
    filename = os.path.join(dirpath, '{}_{}.pth'.format(prefix, epoch))
    torch.save(model.state_dict(), filename)
    print('Saved model to ' + filename)


def save_data(data, prefix, name):
    """
    Save generic data associated with a training run
    :param data: the data to save. The data will be converted to numpy array format.
    :param prefix: the run prefix
    :param name: the name to append to the data file. Should not spaces. ".npy" will be appended to the end of the
                 file automatically.
    """
    dirpath = os.path.join(ROOT_DIR, OUTPUT_DIR)
    makedirs(dirpath)
    filename = os.path.join(dirpath, '{}_{}.npy'.format(prefix, name))
    np.save(filename, np.asarray(data), allow_pickle=False)
    print('Saved data to ' + filename)


def get_timestamp():
    """
    Generate a common-format (ISO) timestamp
    :return: an ISO-format UTC timestamp string.
    """
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


def create_log_dir(prefix):
    timestamp = get_timestamp()
    logdir = os.path.join(ROOT_DIR, OUTPUT_DIR, prefix + "_" + timestamp)
    print('creating logdir ' + str(logdir))
    return logdir


def create_tensorboard_writer():
    return SummaryWriter(log_dir=create_log_dir("tensorboard"))


def exec_sim(sim, zeta, target):
    """
    Calculate the objective function (for minimization) comparing a parameterized simulation rollout
    to a fixed data point.
    :param sim: the simulation to run
    :param zeta: a Python list giving the parameterization to use for the simulation
    :param target: a Python list giving the ball height at each timestep in the original data
    :return:
    """
    estimate = sim.run(zeta)[2]
    diff = np.array(target) - np.array(estimate)
    return np.linalg.norm(diff)


def curry_exec_sim(target, sim):
    """
    Execute a simulator for the given parameter values
    This function simply wraps a simulation and handles the input arguments correctly. It curries (?) the exec_sim
    process, since the function used for CMA evaluation must take a single zeta argument.
    :param sim: the simulator to execute
    :param target: the target output (used to calculate the objective function)
    :return: a list of the objective function values
    """
    def exec_sim_prototype(zeta):
        """ zeta may be a single array of values, or an array of arrays"""
        zeta = [zeta] if np.asarray(zeta).ndim == 1 else zeta  # scalar into list
        zeta = np.asarray(zeta)
        f = [exec_sim(sim, t.tolist(), target) for t in zeta]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    return exec_sim_prototype


def save_files(prefix, objects, filenames):
    assert isinstance(objects, list)
    assert isinstance(filenames, list)
    assert len(objects) == len(filenames)
    for o, f in zip(objects, filenames):
        save_file(prefix, o, f)


def save_file(prefix, object_data, filename):
    outpath = os.path.join(ROOT_DIR, OUTPUT_DIR, prefix, filename)
    makedirs(os.path.dirname(outpath))
    joblib.dump(object_data, outpath)


def load_files(prefix, filenames):
    assert isinstance(filenames, list)
    o = [load_file(prefix, f) for f in filenames]
    return o


def load_file(prefix, filename):
    return joblib.load(os.path.join(ROOT_DIR, OUTPUT_DIR, prefix, filename))


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
