"""
Module containing statistical tests of goodness of fit of conditional density
models.
"""

__author__ = 'Wittawat'

from abc import ABCMeta, abstractmethod
import kcgof

class CGofTest(ABCMeta, object):
    """
    An abstract class for a goodness-of-fit test for conditional density
    models p(y|x). The test requires a paired dataset specified by giving X,
    Y (torch tensors) such that X.shape[0] = Y.shape[0] = n. 
    It is assumed that for each i=1, ..., n,
    Y[i, :] is drawn from r(y|X[i,:]) for some unknown conditional
    distribution r.
    """
    def __init__(self, p, alpha):
        """
        p: UnnormalizedCondDensity 
        alpha: significance level of the test
        """
        self.p = p
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, X, Y):
        """
        X: Torch tensor of size n x dx
        Y: Torch tensor of size n x dy

        perform the goodness-of-fit test and return values computed in a
        dictionary:
        {
            alpha: 0.01, 
            pvalue: 0.0002, 
            test_stat: 2.3, 
            h0_rejected: True, 
            time_secs: ...
        }

        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, X, Y):
        """
        Compute the test statistic. 
        Return a scalar value.
        """
        raise NotImplementedError()

