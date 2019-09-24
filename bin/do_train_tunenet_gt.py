#!/usr/bin/env python


import matplotlib.pyplot as plt
import torch
import torch.utils
import torch.utils.data

from tune.ball_sim import BallSim
from tune.dataset_tunenet import DatasetTuneNet
from tune.model_tunenet import TuneNet
from tune.train_tunenet_gt import INPUT_DIM, OUT_DIM, BATCH_SIZE, device, train, test
from tune.utils import save_model, save_data


def main():
    plt.ion()

    dataset_name_list = ["tunenet_gt",
                         "tunenet_gt_direct",
                         ]
    model_list = [TuneNet(INPUT_DIM, OUT_DIM).to(device),
                  TuneNet(INPUT_DIM, OUT_DIM, degenerate=True).to(device),
                  ]
    train_loader_list = [
        DatasetTuneNet.get_data_loader("tune", "ground_truth", "train", BATCH_SIZE),
        DatasetTuneNet.get_data_loader("tune", "ground_truth", "train", BATCH_SIZE),
    ]
    test_loader_list = [
        DatasetTuneNet.get_data_loader("tune", "ground_truth", "val", BATCH_SIZE),
        DatasetTuneNet.get_data_loader("tune", "ground_truth", "val", BATCH_SIZE),
    ]
    # ensure that the degenerate network directly predicts params instead of differences
    incremental_list = [True,
                        False,
                        ]

    # hyperparameters
    learning_rate = 1e-1
    n_epochs = 10  # PAPER VALUE: 200

    # saving and visualizing
    eval_interval = 1
    save_interval = 100

    for dataset_name, model, train_loader, test_loader, incremental in \
            list(
                zip(dataset_name_list, model_list, train_loader_list, test_loader_list, incremental_list)
            ):
        print("Training TuneNet model on dataset named {}".format(dataset_name))

        # setup PyTorch
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)

        test_err_zeta = []
        test_err_s = []
        train_err_zeta = []
        train_err_s = []

        with BallSim() as sim:
            for epoch in range(1, n_epochs + 1):
                # update learning rate
                scheduler.step()

                # train
                should_eval = epoch % eval_interval == 0 or epoch == n_epochs
                train_err_zeta_i, train_err_s_i = \
                    train(epoch, model, sim, train_loader, optimizer, train_eval_iterations=1,
                          should_eval=should_eval, display_graphs=False, incremental=incremental)

                if should_eval:
                    # only test in this block
                    test_err_zeta_i, test_err_s_i, _, _ = \
                        test(epoch, model, sim, test_loader,
                             tuning_iterations=1, display_graphs=False, incremental=incremental)
                    train_err_zeta.append(train_err_zeta_i)
                    train_err_s.append(train_err_s_i)
                    test_err_zeta.append(test_err_zeta_i)
                    test_err_s.append(test_err_s_i)

                # test
                if epoch % save_interval == 0 or epoch == n_epochs:
                    save_model(model, epoch, dataset_name)

                # save
            save_data(train_err_zeta, dataset_name, "train_err_zeta")
            save_data(train_err_s, dataset_name, "train_err_s")
            save_data(test_err_zeta, dataset_name, "test_err_zeta")
            save_data(test_err_s, dataset_name, "test_err_s")


if __name__ == "__main__":
    main()
