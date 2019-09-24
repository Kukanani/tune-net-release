#!/usr/bin/env python

import sys

from tune.ball_sim import BallSim

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("You must supply a type of dataset to generate.")
        sys.exit(1)

    # print("saving every {} frames".format(save_skip_frames))
    # print("{} datapoints per run".format(num_datapoints))

    if sys.argv[1] == "generate_tune_gt":
        with BallSim() as bs:
            bs.generate_tune_dataset("tune", plane_jitter=0.0, random_render=False, render=False)

    elif sys.argv[1] == "generate_tune_gt_hard":
        with BallSim() as bs:
            bs.generate_tune_dataset("tune_hard", plane_jitter=0.0, random_render=False, render=False,
                                     range_override=(0.0, 1.0))

    elif sys.argv[1] == "generate_tune_obs":
        with BallSim() as bs:
            bs.generate_tune_dataset("tune_dynview", plane_jitter=0.0, random_render=True, render=True)

    elif sys.argv[1] == "generate_tune_obs_hard":
        with BallSim() as bs:
            bs.generate_tune_dataset("tune_dynview_hard", plane_jitter=0.0, random_render=True, render=True,
                                     range_override=(0.0, 1.0))
    else:
        print("that's not a dataset type I know how to generate.")
