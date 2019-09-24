#!/usr/bin/env python
#

#
# Basic definitions and constants. This file can be modified to aid portability
# to different machines/setups (i.e. changing search paths).

from pathlib import Path

##############################################################

ROOT_DIR = Path(__file__).parents[1].absolute().as_posix()

# relative to ROOT_DIR
DATASET_DIR = "datasets"

# relative to ROOT_DIR
OUTPUT_DIR = "saves"
