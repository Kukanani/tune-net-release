#!/usr/bin/env python
# coding: utf-8
#

#
# Tune an environment using greedy entropy search.
#
# The entropy search method was first proposed in the following paper:
#   Hennig, Philipp, and Schuler, Christian J. "Entropy search for information-efficient
#   global optimization." JMLR 2012.
#   http://www.jmlr.org/papers/v13/hennig12a.html
#
# Implementation details in this file are based on parts of the following paper:
#   Zhu, Shaojun and Kimmel, Andrew and Bekris, Kostas E. and Boularias, Abdeslam.
#   "Fast Model Identification via Physics Engines for Data-Efficient Policy Search."
#   IJCAI 2018.
#   https://arxiv.org/abs/1710.08893,
#   https://www.ijcai.org/proceedings/2018/451,
#   https://doi.org/10.24963/ijcai.2018/451
#
import numpy as np
import GPy
import math
import torch

def fit_gp(x, y):
    """
    Fit a gp to the provided data and calculate max-likelihood parameters
    :param x: x points of the data
    :param y: y points of the data
    :return: an optimized GP object.
    """
    # Use GPy
    kernel = GPy.kern.RBF(1)
    #     print(np.asarray(x))
    #     print(np.asarray(y))
    model = GPy.models.GPRegression(np.asarray(x), np.asarray(y), kernel)
    model.optimize()

    return model


def sample_gp(model, n, x_pred):
    """
    Sample n functions from the GP.
    :param model: the Gaussian process to sample mean/covariance from and sample at x_pred.
    :param n: the number of functions to sample
    :param x_pred: the positions to use for GP sampling.
    :return: a list of python dictionaries with keys = x_pred
    """
    E_list = []
    x = np.expand_dims(np.array(x_pred), -1)

    samps = model.posterior_samples(x, size=n)
    samps_collated = np.moveaxis(np.squeeze(samps), 1, 0)
    for j in range(n):
        E_list.append({(x_pred[i]): samps_collated[j][i] for i in range(len(x_pred))})

    return E_list


def entsearch(buffer, sim, big_theta, epsilon, k_max, n):
    """
    Main VGMI algorithm. This function seeks to replicate Algorithm 2 of the source paper.

    :param P: the probability distribution to update
    :param buffer: a buffer of data from the target environment: [zeta_t, [s_0, s_1, ..., s_t]]
    :param big_theta: discretized space (Python list) of possible physical property values
    :param sim: the simulator to use for evaluation. Must support sim.run(theta)
    :param epsilon: if theta_k doesn't change by more than this amount, stop iterating.
    :param k_max: maximum # of evaluated models
    :param n: number of error functions to MC sample
    :return: history of probability distribution P over big_theta, in the form
             [{big_theta_0: p(big_theta_0), big_theta_1: P(big_theta_1), ... },
              {...},
              {...}], one dict for each time step
    """
    # starting value for theta_k
    theta_k = np.random.choice(big_theta)
    # buffer used to learn GP
    P_history = [{theta: 1 / len(big_theta) for theta in big_theta}]
    L_theta = []
    L_loss = []
    last_best = 0
    for k in range(k_max):
        # calculating the accuracy of model theta_k
        P_k = {}

        estimate = sim.run([theta_k, buffer[0][1]])[2]
        diff = np.array(buffer[1]) - np.array(estimate)
        l_k = np.linalg.norm(diff)

        L_theta.append([theta_k])
        L_loss.append([l_k])
        gp = fit_gp(np.asarray(L_theta), np.asarray(L_loss))

        # Monte Carlo sampling
        # Sample E_list = [E_1, E_2, ..., E_n] from GP(m, K) in big_theta
        # Each E is a dictionary with keys = big_theta
        E_list = sample_gp(gp, n, big_theta)

        for theta in big_theta:
            indicator_sum = 0
            for E in E_list:
                min_E = min(E, key=E.get)
                if theta == min_E:
                    indicator_sum += 1
            P_k[theta] = 1 / n * indicator_sum

        entropy = {theta: P_k[theta] * math.log(P_k[theta], 2) if P_k[theta] > 0 else 100 for theta in big_theta}
        theta_k = min(entropy, key=entropy.get)
        P_history.append(P_k)
        best = max(P_k, key=P_k.get)
        if abs(best - last_best) < epsilon:
            # stop early
            for kt in range(k+1, k_max):
                P_history.append(P_k)
            break
        last_best = best

    # best_estimates = []
    # best_so_far = 0
    # for p in P_history:
    #     best_so_far = max([best_so_far, max(p, key=p.get)])
    #     best_estimates.append(best_so_far)

    best_estimates = [max(p, key=p.get) for p in P_history]
    # have to set the first best estimate to a random one, since it will default
    # to the start of the range because all the probabilities are equal.
    best_estimates[0] = np.random.choice(list(P_history[0].keys()))
    # print(best_estimates)
    return P_history, best_estimates


def entsearch_over_dataset(loader, sim, *args, **kwargs):
    num = len(loader.dataset)
    p_history = []
    estimates = []
    targets = np.empty([num])

    with torch.no_grad():
        batch_size = 0
        for batch_idx, (zeta_batch, s_batch, v_batch) in enumerate(loader):
            if batch_idx == 0:
                batch_size = zeta_batch.shape[0]
            for idx_in_batch in range(zeta_batch.shape[0]):
                # add one to this index because the 0 index is the starting point for the algorithm
                idx = batch_idx * batch_size + idx_in_batch

                print("entropy search, dataset idx " + str(idx))
                zeta = zeta_batch[idx_in_batch].float().cpu().numpy()
                target_zeta = zeta[1].tolist()
                state = sim.run(target_zeta)[2]
                buffer = [target_zeta, state]

                this_history, this_estimate = entsearch(buffer, sim, *args, **kwargs)
                p_history.append(this_history)
                estimates.append(this_estimate)
                targets[idx] = target_zeta[0]
    return p_history, np.asarray(estimates), np.asarray(targets)
