"""
A module containing utility functions.
Prefix pt_ for functions operating on Pytorch tensors.
"""

# all utility functions in kgof.util are visible.
from kgof.util import *

import numpy as np
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

def pt_sample_standard_normal(n, d):
    """
    Generate samples from the standard normal. Return a torch tensor.
    """
    return torch.randn(n,d)

def pt_dist2_matrix(X, Y=None):
    """
    Construct a pairwise Euclidean distance **squared** matrix of size
    X.shape[0] x Y.shape[0]

    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (X**2).sum(1).view(-1, 1)
    if Y is not None:
        y_norm = (Y**2).sum(1).view(1, -1)
    else:
        Y = X
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    # Some entries can be very small negative
    dist[dist <= 0] = 0.0
    return dist

def pt_meddistance(X, subsample=None, seed=283):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d torch tensor

    Return
    ------
    median distance (a scalar, not a torch tensor)
    """
    n = X.shape[0]
    if subsample is None:
        D = torch.sqrt(pt_dist2_matrix(X, X))
        I = torch.tril_indices(n, n, -1)
        Tri = D[I[0], I[1]]
        med = torch.median(Tri)
        return med.item()
    else:
        assert subsample > 0
        with NumpySeedContext(seed=seed):
            ind = np.random.choice(n, min(subsample, n), replace=False)
        # recursion just once
        return pt_meddistance(X[ind], None, seed=seed)

