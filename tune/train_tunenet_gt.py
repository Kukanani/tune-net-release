#!/usr/bin/env python
#

#
# Train TuneNet on position-position bouncing ball data, which is very similar to the dataset of Ajay et al 2018.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils
import torch.utils.data
import torch.utils.data

from tune.utils import get_torch_device, create_tensorboard_writer

device = get_torch_device()
writer = create_tensorboard_writer()

TIME_LENGTH = 400
SERIES_COUNT = 2
INPUT_DIM = TIME_LENGTH
OUT_DIM = 1
BATCH_SIZE = 50

ax = None
loss_fn = torch.nn.MSELoss()


def train(epoch,
          model,
          sim,
          data_loader,
          optimizer,
          train_eval_iterations,
          should_eval=False,
          display_graphs=False,
          incremental=True):
    train_loss = 0
    batch_idx = 0
    for (zeta_batch, s_batch, _) in data_loader:
        zeta_batch = zeta_batch.float().to(device)
        s_batch = s_batch.float().to(device).permute(0, 2, 1)
        input_i = torch.tensor(
            np.reshape(s_batch[:, :, :SERIES_COUNT].cpu(), ([-1, TIME_LENGTH * SERIES_COUNT]), order="F")).to(device)
        input_i.requires_grad = True

        # TODO: the naming on delta_zeta_batch is misleading. If the network is not incremental, this is
        #       not delta_zeta, but just zeta.
        if incremental:
            delta_zeta_batch = zeta_batch[:, 1].sub(zeta_batch[:, 0])
        else:
            delta_zeta_batch = zeta_batch[:, 1]

        delta_zeta_hat = model(input_i).squeeze()
        delta_zeta = delta_zeta_batch[:, 0].squeeze()

        optimizer.zero_grad()

        loss = loss_fn(delta_zeta_hat, delta_zeta)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        batch_idx += 1

    err_s = None
    err_zeta = None
    print('====> Epoch: {} Average loss: {}'.format(
        epoch, train_loss / len(data_loader.dataset)))
    if should_eval:
        err_zeta, err_s, _, _ = test(epoch,
                                     model,
                                     sim,
                                     data_loader,
                                     train_eval_iterations,
                                     display_graphs,
                                     test_type="train",
                                     incremental=incremental)

    return err_zeta, err_s


def test(epoch, model, sim, data_loader, tuning_iterations=1, display_graphs=False, test_type="test", incremental=True):
    """
    Perform tests over a dataset to calculate the error in parameter and simulated state
    :param epoch: the epoch at evaluation time (used for tensorboard logging
    :param model: the model to use for evaluation
    :param tuning_iterations: number of tuning iterations to perform
    :param display_graphs: if True, display graphs after tuning showing some examples
    :param data_loader: the ground truth data to test over
    :param test_type: a string describing why this test is run. The tests can be run during training, which
                      can make printouts confusing, so this string is used to disambiguate.
    :param incremental: if True, then each iteration will add to the starting value (the model estimates the difference)
                        rather than simply setting the value (the model estimates the value).
    :return:
    """
    print("Testing over " + test_type + "...")

    dataset_size = len(data_loader.dataset)
    print('dataset size is ' + str(dataset_size))
    s = torch.zeros((dataset_size, TIME_LENGTH, 2)).to(device)
    v = torch.zeros((dataset_size, TIME_LENGTH, 2)).to(device)
    s_hat = torch.zeros((dataset_size, TIME_LENGTH)).to(device)
    v_hat = torch.zeros((dataset_size, TIME_LENGTH)).to(device)

    zeta_list = torch.zeros((dataset_size, 1)).to(device)
    zeta_hat_history_list = torch.zeros((dataset_size, tuning_iterations + 1)).to(device)

    # generate predictions
    with torch.no_grad():
        # count = 0
        for batch_idx, batch_data in enumerate(data_loader):
            zeta_batch = batch_data[0]
            s_batch = batch_data[1]
            if len(batch_data) > 2:
                v_batch = batch_data[2]
            for idx_in_batch in range(zeta_batch.shape[0]):
                # print(idx_in_batch)
                idx = batch_idx * BATCH_SIZE + idx_in_batch
                # print(idx)
                # pull out the first datapoint from the batch.
                zeta_i = zeta_batch[idx_in_batch].float()
                s[idx] = s_batch.float().to(device).permute(0, 2, 1)[idx_in_batch]
                if len(batch_data) > 2:
                    v[idx] = v_batch.float().to(device).permute(0, 2, 1)[idx_in_batch]

                # extract relevant physics information from datapoint.
                # zeta_i is a vstack. Row 1 is the source sim, row 2 is the target sim
                # each row is a list of all the physics params, such as (restitution, drop_height).
                # print(zeta_i)
                zeta_list[idx] = zeta_i[1, 0]

                zeta_hat_history_list[idx, :], s_hat[idx, :], v_hat[idx, :] = \
                    tune_iter(model, sim, s, zeta_i, idx, tuning_iterations, display_graphs,
                              incremental=incremental)
                # print("zeta_list evolution: " + str(zeta_hat_history_list[idx, :]))

                # count += 1
        # print("{} datapoints processed.".format(count))

    err_s = torch.abs(s[:, :, 1] - s_hat[:, :]).cpu().numpy()
    # err_s_percentage = np.abs(np.divide(err_s, s[:, :, 1].cpu().numpy() + 0.0000001) * 100.)
    # err_v = torch.abs(v[:, :, 1] - v_hat[:, :]).cpu().numpy()

    # compare the last iteration of zeta_hat_history_list with zeta_list to compute the mean absolute error
    err_zeta = torch.abs(zeta_list - zeta_hat_history_list).cpu().numpy()
    last_err_zeta = err_zeta[:, -1]
    # writer.add_scalar('{}_mae_s'.format(test_type), np.mean(err_s, keepdims=False), epoch)
    writer.add_scalar('{}_mae_zeta'.format(test_type), np.mean(last_err_zeta, keepdims=False), epoch)
    print("mae of zeta_list: {:6.4f}".format(np.mean(last_err_zeta, keepdims=False)))
    print("mse of zeta_list: {:f}".format(np.mean(last_err_zeta * last_err_zeta, keepdims=False)))

    return np.mean(err_zeta, keepdims=False), np.mean(err_s, keepdims=False), zeta_hat_history_list, err_zeta


def tune_iter(model, sim, s_start, zeta_start, idx, tuning_iterations,
              display_graphs, incremental):
    assert tuning_iterations > 0
    # zeta = zeta_start.clone()
    s = s_start.clone()
    position_list = linear_velocity_list = None
    zeta_hat_history = torch.tensor(np.zeros([tuning_iterations + 1]))
    zeta_hat_history[0] = zeta_start[0, 0]

    # print("starting zeta value: " + str(zeta_start[0, 0]))
    for iters in range(tuning_iterations):
        input_i = torch.tensor(
            np.reshape(s[idx, :, :SERIES_COUNT].cpu(), ([-1, TIME_LENGTH * SERIES_COUNT]), order="F")).to(device)

        delta_zeta_hat = model(input_i).item()

        # calculate new parameters
        previous_zeta = zeta_hat_history[iters]

        if incremental:
            new_zeta = previous_zeta + delta_zeta_hat
        else:
            new_zeta = delta_zeta_hat
        new_zeta = max(0, new_zeta)

        if tuning_iterations == 1:
            # special case to speed things up: don't run the sim
            position_list = torch.zeros(TIME_LENGTH, 3)
            linear_velocity_list = torch.zeros(TIME_LENGTH, 3)
            zeta_hat_history[iters + 1] = new_zeta
            return zeta_hat_history, torch.tensor(position_list[:, 2]), torch.tensor(linear_velocity_list[:, 2])

        # get new rollout
        obj_pos = [0, 0, zeta_start[1, 1]]
        _, _, position_list, _, linear_velocity_list, _, _, _, _ = sim.run(zeta=[new_zeta, obj_pos], render=False)
        if display_graphs:
            # do_display(input_i, position_list, zeta_start, new_zeta, s[idx])
            pass

        s[idx, :, 0] = torch.tensor(position_list[:, 2])
        zeta_hat_history[iters + 1] = new_zeta

    # print("zeta hat history: " + str(zeta_hat_history))
    return zeta_hat_history, torch.tensor(position_list[:, 2]), torch.tensor(linear_velocity_list[:, 2])


def do_display(input_i, position_list, zeta_target, zeta_hat, s_i):
    global ax
    if ax is None:
        _, ax = plt.subplots(2, 1)
    ax[0].cla()
    ax[0].set_ylim([0, 5])
    ax[1].cla()
    ax[1].set_ylim([0, 5])
    # note: rho is the symbol for COR, but this should be changed if the semantic meaning
    # of the zeta parameter changes.
    ax[0].plot(position_list[:, 2], label="approximate run 1 (rho={:.4f})".format(zeta_hat),
               color=(0.0, 0.5, 0.0, 1.0), ls="dashed")
    ax[1].plot(input_i[0, :].detach().cpu().squeeze().numpy())

    ax[0].plot(s_i[:, 0].detach().squeeze().cpu().numpy(),
               label="actual run 0 (rho={:.4f})".format(zeta_target[0, 0]), color=(0.0, 0.0, 0.0))
    ax[0].plot(s_i[:, 1].detach().squeeze().cpu().numpy(),
               label="actual run 1 (rho={:.4f})".format(zeta_target[1, 0]), color=(0.0, 0.5, 0.0))

    ax[0].legend()

    plt.pause(1e-6)
