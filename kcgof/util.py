# all utility functions in kgof.util are visible.
from kgof.util import *

import torch

class TorchSeedContext(object):
    """
    A context manager to reset the random seed used by torch.randXXX(...)
    Set the seed back at the end of the block. 
    """
    def __init__(self, seed):
        self.seed = seed 

    def __enter__(self):
        rstate = torch.get_rng_state()
        self.cur_state = rstate
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *args):
        torch.set_rng_state(self.cur_state)

# end TorchSeedContext


