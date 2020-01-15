This repository contains code for the paper:

"TuneNet: One-Shot Residual Tuning for System Identification and Sim-to-Real Robot Task Transfer"

Adam Allevato, Elaine Schaertl Short, Mitch Pryor, and Andrea Thomaz

CoRL 2019

https://arxiv.org/abs/1907.11200

This includes all code needed to make the plots contained in the paper (Figure 2), showing parameter error
as a function of simulation rollouts.

# Video
[![TuneNet Video](https://i.imgur.com/XyZuax1.png)](https://youtu.be/Ux2pSmZx3uE "TuneNet Video")

# Requirements

- Python 3.5+
- Conda

# Reading instructions
TuneNet is implemented using PyTorch.
The network architecture is defined in `tune/model_tunenet.py`.
The network train and test code is in `tune/train_tunenet_gt.py`.

# Running instructions
Some numbers (# of tuning iterations, # of training epochs) have been modified for faster runtime, at the cost of reduced performance. To achieve full performance, search the codebase for "PAPER VALUE" and update the defined variables to their original values as specified in the comments.

1. Import the conda environment 
```
conda env create -f environment.yml
conda activate tunenet
pip install -e .
```

1. Generate bouncing-ball dataset
```
python bin/generate_ball_dataset.py generate_tune_gt
```

1. Train TuneNet and TuneNet "Direct Prediction" models
```
python bin/do_train_tunenet_gt.py
```
Tensorboard runs are automatically saved to the `saves` directory.

1. Install kernel and launch Jupyter notebook 
```
python -m ipykernel install --user --name tunenet --display-name "Python (tunenet)"
jupyter notebook
```

1. Open `notebooks/plot_tunenet_performance.ipynb`

1. switch to the new kernel

1. Run the entire notebook to generate performance plots and compare with other gradient-free estimation techniques


