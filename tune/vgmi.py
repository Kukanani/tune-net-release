#!/usr/bin/env python
#
# Implementation of Value-Guided Model Identification (VGMI) in Python.
# The approach is from "Fast Model Identification via Physics Engines
# for Data-Efficient Policy Search", Shaojun Zhu, Andrew Kimmel,
# Kostas E. Bekris, Abdeslam Boularias, IJCAI 2018.
# https://arxiv.org/abs/1710.08893
#

import random

import math
import numpy as np

import GPy

# import george
# from george import kernels
# from scipy.optimize import minimize

# Constants: gotta define them somewhere
# TODO: tune these values, right now they are complete guesses
BIG_THETA = np.linspace(start=0, stop=1, num=21)
KMIN = 5
KMAX = 50
ETA = 0.01
EPSILON = 0.01
N = 10
# number of timesteps in a single data collection rollout
H = 400


def fit_gp(x, y):
    """
    Fit a gp to the provided data and calculate max-likelihood parameters
    :param x: x points of the data
    :param y: y points of the data
    :return: an optimized GP object.
    """
    # Use George
    # # TODO: choose a kernel
    # kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
    # gp = george.GP(kernel)
    # gp.compute(x, y)
    #
    # def neg_ln_like(p):
    #     gp.set_parameter_vector(p)
    #     return -gp.log_likelihood(y)
    #
    # def grad_neg_ln_like(p):
    #     gp.set_parameter_vector(p)
    #     return -gp.grad_log_likelihood(y)
    #
    # result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)
    # return gp, result.x

    # Use GPy
    kernel = GPy.kern.RBF(7)
    model = GPy.models.GPRegression(x, y, kernel)
    model.optimize()

    return model


def sample_gp(model, n, x_pred):
    """
    Sample n functions from the GP.
    :param model: the Gaussian process to sample mean/covariance from and sample at x_pred.
    :param n: the number of functions to sample
    :return: a list of python dictionaries with keys = x_pred
    """
    E_list = {}
    x = np.expand_dims(np.array(x_pred), -1)

    samps = model.posterior_samples(x, size=n)
    samps_collated = np.moveaxis(np.squeeze(samps), 1, 0)
    for j in range(n):
        E_list.append({tuple(x_pred[i]): samps_collated[j][i] for i in range(len(x_pred))})

    return E_list


def simulate_one_tick(theta, state, action):
    """
    Simulate one timestep of the simulation.

    :param theta: the parameters to use for the simulation
    :param state: the state to start the simulation in
    :param action: the action to take at this timestep
    :return: the new state after taking this action
    """
    # TODO
    return new_state


def trpo(pi, theta):
    """
    Perform TRPO to improve a policy in a simulation.
    :param pi: the policy to begin with
    :param theta: the parameters to use for the simulation
    :return: an improved policy
    """
    # TODO
    return pi_prime


def get_value_of_policy(pi, theta):
    """
    Evaluate a policy over a parameterized simulation.
    :param pi: the policy to execute
    :param theta: the simulation parameterization to evaluate over
    :return: the expected value of executing policy pi in theta-parameterized simulation
    """
    # TODO
    return v

def vgmi(P, sas, big_theta, pi, k_min, k_max, eta, epsilon, n):
    """
    Main VGMI algorithm. This function seeks to replicate Algorithm 2 of the source paper.

    :param P: the probability distribution to update
    :param sas: state-action-state data in the form [(x_i, mu_i, x_{i+1})_{i=0}, ... ){i=t}]
    :param big_theta: discretized space (Python list) of possible physical property values
    :param pi: reference policy
    :param k_min: minimum # of evaluated models
    :param k_max: maximum # of evaluated models
    :param eta: model confidence threshold
    :param epsilon: value error threshold
    :param n: number of error functions to MC sample
    :return: probability distribution P over big_theta, in the form
             {big_theta_0: p(big_theta_0), big_theta_1: P(big_theta_1), ... }
    """

    # initalize some variables (some not in algo listing)

    # starting value for theta_k
    theta_k = random.choice(big_theta)
    # buffer used to learn GP
    L = []
    # loop counter
    k = 0
    # loop control
    stop = False

    while not stop:

        # calculating the accuracy of model theta_k
        l_k = 0
        for sas_tuple in sas:
            # simulate {(x_i, mu_i)} using physics engine with params theta_K and get
            # predicted next state xhat_next = xhat_{i+1} = f(x_i, mu_i, theta_k)
            xhat_next = simulate_one_tick(theta_k, sas_tuple[0], sas_tuple[1])
            l_k = l_k + np.linalg.norm((xhat_next - sas_tuple[2]))

        L.append((theta_k, l_k))
        gp = fit_gp(list(zip(*L)))

        # Monte Carlo sampling
        # Sample E_list = [E_1, E_2, ..., E_n] from GP(m, K) in big_theta
        # Each E is a dictionary with keys = big_theta
        E_list = sample_gp(gp, n, big_theta)
        for theta in big_theta:
            indicator_sum = 0
            for E in E_list:
                min_E = min([E[theta_prime] for theta_prime in big_theta])
                if theta == min_E:
                    indicator_sum += 1
            P[theta] = 1/n * indicator_sum

        # Selecting the next model to evaluate
        # Note that this is the greedy selection mechanism. Equivalent to Eqn (4)
        entropy = [P[theta]*math.log(P[theta], 2) for theta in big_theta]
        theta_k = min(entropy, key=entropy.get)

        # checking the stopping condition
        if k >= k_min:
            theta_star = big_theta[P.index(max(P))]
            # Calculate the values V^pi(theta) with all models theta that have a
            # probability P(theta) > eta by using the physics engine for
            # simulating trajectories with models theta
            val_sum = 0
            for theta in big_theta:
                if P[theta] >= eta:
                    val_sum += P[theta] * abs(get_value_of_policy(pi, theta) - get_value_of_policy(pi, theta_star))
                else:
                    pass
            if val_sum <= epsilon:
                stop = True

        if k > k_max:
            stop = True
    return P


def main_loop():
    """
    Recreate Algorithm 1 from the source paper.
    :return:
    """
    t = 0
    # probability over big_theta (return value)
    big_theta = BIG_THETA
    P = {theta: 1/len(big_theta) for theta in big_theta}
    # TODO: Initialize policy pi

    # TODO: remove this loop counter
    loop_count = 0
    while True:
        sas = []
        # TODO: Execute policy pi for H iterations and collect new state-action-state data
        #       for i = t, ..., t+H-1, fill sas variable
        t = t + H
        P = vgmi(P=P,
                 sas=sas,
                 big_theta=big_theta,
                 pi=pi,
                 k_min=KMIN,
                 k_max=KMAX,
                 eta=ETA,
                 epsilon=EPSILON,
                 n=N)
        # initialize a policy search algorithm (e.g TRPO) with pi and run the algo
        # in the simulator with the model theta = max(P, key=P.get) to find improved
        # policy pi_prime
        pi = trpo(pi, max(P, key=P.get))

        # TODO: replace this loop count logic with a stopping condition based on time
        loop_count += 1
        print("completed loop " + str(loop_count))
        if loop_count > 10:
            break


def main():
    main_loop()


if __name__ == "__main__":
    main()
