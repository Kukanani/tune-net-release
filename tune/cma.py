#!/usr/bin/env python
# coding: utf-8
#

#
# tune a simulator's parameters using CMA-ES optimization.
import os

import cma
import numpy as np
import torch
from itertools import zip_longest

from tune.utils import create_log_dir, curry_exec_sim


def do_cma(target, sim, initial_guess, run_name=None, **kwargs):
    if run_name is None:
        run_name = create_log_dir("cmaes")

    my_exec_sim = curry_exec_sim(target, sim)

    inopts = {
        "tolfun": 0.1,
        "verb_disp": 1,
        "verb_filenameprefix": run_name + "/",
        # "verbose": -1,
    }
    inopts.update(kwargs)
    es = cma.CMAEvolutionStrategy(initial_guess, 0.1, inopts=inopts)
    es.optimize(my_exec_sim)

    result = es.result
    evals = es.result.evals_best

    output = load_cma_output(run_name)

    return result, evals, output


def load_cma_output(run_name):
    import csv

    data = []
    reader = csv.reader(open(os.path.join(run_name, "xmean.dat")), delimiter=" ")
    for idx, row in enumerate(reader):
        if idx > 0:
            data.append([float(r) for r in row])
    return np.asarray(data)


def do_cma_over_dataset(loader, sim, maxfevals, popsize):
    print("newstyle")
    num = len(loader.dataset)
    estimate_dims = 2
    cma_sim_evals = []
    cma_estimates = []
    cma_targets = np.empty([num, estimate_dims])

    with torch.no_grad():
        batch_size = 0
        for batch_idx, (zeta_batch, s_batch, v_batch) in enumerate(loader):
            if batch_idx == 0:
                batch_size = zeta_batch.shape[0]
            for idx_in_batch in range(zeta_batch.shape[0]):
                # add one to this index because the 0 index is the starting point for the algorithm
                idx = batch_idx * batch_size + idx_in_batch

                zeta = zeta_batch[idx_in_batch].float().cpu().numpy()
                cma_starting_guess = zeta[0].tolist()
                target_zeta = zeta[1].tolist()
                target_observation = sim.run(target_zeta)[2]
                _, _, cma_output = do_cma(target_observation, sim, cma_starting_guess, run_name=None,
                                          maxfevals=maxfevals, popsize=popsize)

                cma_sim_eval = [0]
                cma_sim_eval.extend(cma_output[:, 1].tolist())
                cma_sim_evals.append(cma_sim_eval)

                cma_estimate = [cma_starting_guess]
                cma_estimate.extend(cma_output[:, 5:].tolist())
                cma_estimates.append(cma_estimate)

                cma_targets[idx] = target_zeta
    lengths = [len(e) for e in cma_estimates]
    if min(lengths) != max(lengths):
        print("ERROR: CMAES runs are of different iteration lengths! Need to add padding code")
        # make all runs the same length for averaging purposes
        cma_estimates = pad_length(cma_estimates)
        cma_sim_evals = pad_length(cma_sim_evals)
    cma_estimates = np.stack(cma_estimates, axis=0)
    cma_sim_evals = np.stack(cma_sim_evals, axis=0)

    return cma_sim_evals, cma_estimates, cma_targets


def pad_length(arr):
    # this is so ugly. So so ugly.
    # print(arr)
    max_subarr_length = max([len(a) for a in arr])
    # print([len(a) for a in arr])

    arr = [np.asarray(a) for a in arr]
    def pad_amount(a):
        amt = [(0, max_subarr_length - a.shape[0])]
        for i in range(a.ndim-1):
            amt.extend([(0, 0)])
        # print(amt)
        return amt
    new_arr = [np.pad(a, pad_amount(a), 'edge') for a in arr]
    # print([a.shape for a in new_arr])
    # convert to a single numpy array
    # print(new_arr)
    new_arr = np.stack(new_arr, axis=0)
    # print(new_arr)
    return new_arr
