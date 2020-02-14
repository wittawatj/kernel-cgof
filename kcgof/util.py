"""
A module containing utility functions.
Prefix pt_ for functions operating on Pytorch tensors.
"""

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


# def pt_dist2_matrix(X, Y):
#     """
#     Construct a pairwise Euclidean distance **squared** matrix of size
#     X.shape[0] x Y.shape[0]

#     https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
#     """
#     sx = np.sum(X**2, 1)
#     sy = np.sum(Y**2, 1)
#     D2 =  sx[:, np.newaxis] - 2.0*np.dot(X, Y.T) + sy[np.newaxis, :] 
#     return D2
