
"""
This file defines global configuration of the project.
Casual usage of the package should not need to change this. 
"""

import kcgof.glo as glo
import os
import torch

# All keys with prefix ex_ are only relevant for batch experiments i.e.,
# relevant to experiment scripts under kcgof/ex/
_default_config = {
    #  default torch data type
    'torch_dtype': torch.double,

    # default torch device
    'torch_device': torch.device('cpu'),

    # Full path to the directory to store temporary files when running
    # experiments.     
    'ex_scratch_path': '/is/ei/wittawat/tmp/kcgof',

    # Slurm partitions.
    # When using SlurmComputationEngine for running the experiments, the partitions (groups of computing nodes)
    # can be specified here. Set to None to not set to any value (i.e., use the default partition).
    # The value is a string. For more than one partition, set to, for instance, "wrkstn,compute".
    'ex_slurm_partitions': None,

    # Full path to the directory to store experimental results.
    'ex_results_path': '/is/ei/wittawat/results/kcgof',

    # Full path to the root directory of the shared folder. This folder contains
    # all resource files (e.g., data, trained models) that are released by the
    # authors. 
    'shared_resource_path': '/is/ei/wittawat/Gdrive/kcgof_share/',
}

def set_default_config(config):
    global _default_config
    _default_config = config

def get_default_config():
    # Keys with prefix ex_ are used only when scripts under kcgof/ex/ are
    # executed. 
    return _default_config