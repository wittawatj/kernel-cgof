"""
Module containing implementation of conditional density functions.
"""

__author__ ='Wittawat'

from abc import ABCMeta, abstractmethod
import scipy.stats as stats
import kcgof
import kcgof.log as log
#import warnings

def warn_bounded_domain(self):
    log.l().warning('{} has a bounded domain. This may have an unintended effect to the test result'.format(self.__class__) )

# def from_log_den(d, f):
#     """
#     Construct an UnnormalizedDensity from the function f, implementing the log 
#     of an unnormalized density.

#     f: X -> den where X: n x d and den is a numpy array of length n.
#     """
#     return UDFromCallable(d, flog_den=f)

# def from_grad_log(d, g):
#     """
#     Construct an UnnormalizedDensity from the function g, implementing the
#     gradient of the log of an unnormalized density.

#     g: X -> grad where X: n x d and grad is n x d (2D numpy array)
#     """
#     return UDFromCallable(d, fgrad_log=g)


class UnnormalizedCondDensity(ABCMeta, object):
    """
    An abstract class of an unnormalized conditional probability density
    function. This is intended to be used to represent a condiitonal model of
    the data    for goodness-of-fit testing. Specifically, the class specifies

    p(y|x) where the normalizer may not be known. That is, in fact, it
    specifies p(y,x) since p(y|x) = p(y,x)/p(x), and p(x) is the normalizer.
    The normalizer of the joint density is not assumed to be known.
    """

    @abstractmethod
    def log_den(self, X, Y):
        """
        Evaluate log of the unnormalized density on the n points in (Y, X)
        i.e., log p(y_i, x_i) (up to the normalizer) for i = 1,..., n.
        Y, X are paired.

        X: n x dx Torch tensor
        Y: n x dy Torch tensor

        Return a one-dimensional Torch array of length n.
        """
        raise NotImplementedError()

    # def log_normalized_den(self, X):
    #     """
    #     Evaluate the exact normalized log density. The difference to log_den()
    #     is that this method adds the normalizer. This method is not
    #     compulsory. Subclasses do not need to override.
    #     """
    #     raise NotImplementedError()

    def get_condsource(self):
        """
        Return a CondSource that allows sampling from this density.
        May return None if no CondSource is implemented.
        Implementation of this method is not enforced in the subclasses.
        """
        return None

    def grad_log(self, X, Y):
        """
        Evaluate the gradients (with respect to Y) of the conditional log density at
        each of the n points in X. That is, compute 
        
        grad_yi log p(y_i | x_i) 

        and stack all the results for i =1,..., n. 
        Assume X.shape[0] = Y.shape[0] = n.
        
        Given an implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.

        X: n x dx Torch tensor
        Y: n x dy Torch tensor

        Return an n x dy Torch array of gradients.
        """
        # g = autograd.elementwise_grad(self.log_den)
        # G = g(X)
        # return G
        raise NotImplementedError()

    @abstractmethod
    def dy(self):
        """
        Return the dimension of Y.
        """
        raise NotImplementedError()

    @abstractmethod
    def dx(self):
        """
        Return the dimension of X.
        """
        raise NotImplementedError()

# end UnnormalizedCondDensity


# class UDFromCallable(UnnormalizedDensity):
#     """
#     UnnormalizedDensity constructed from the specified implementations of 
#     log_den() and grad_log() as callable objects.
#     """
#     def __init__(self, d, flog_den=None, fgrad_log=None):
#         """
#         Only one of log_den and grad_log are required.
#         If log_den is specified, the gradient is automatically computed with
#         autograd.

#         d: the dimension of the domain of the density
#         log_den: a callable object (function) implementing the log of an unnormalized density. See UnnormalizedDensity.log_den.
#         grad_log: a callable object (function) implementing the gradient of the log of an unnormalized density.
#         """
#         if flog_den is None and fgrad_log is None:
#             raise ValueError('At least one of {log_den, grad_log} must be specified.')
#         self.d = d
#         self.flog_den = flog_den
#         self.fgrad_log = fgrad_log

#     def log_den(self, X):
#         flog_den = self.flog_den
#         if flog_den is None:
#             raise ValueError('log_den callable object is None.')
#         return flog_den(X)

#     def grad_log(self, X):
#         fgrad_log = self.fgrad_log
#         if fgrad_log is None:
#             # autograd
#             g = autograd.elementwise_grad(self.flog_den)
#             G = g(X)
#         else:
#             G = fgrad_log(X)
#         return G

#     def dim(self):
#         return self.d

# end UDFromCallable

    