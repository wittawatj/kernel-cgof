"""
Module containing implementation of conditional density functions.
"""

__author__ ='Wittawat'

from abc import ABCMeta, abstractmethod
import scipy.stats as stats
import kcgof
import kcgof.log as log
import kcgof.glo as glo
import kcgof.cdata as cdat
import torch
import torch.distributions as dists
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


class UnnormalizedCondDensity( object):
    """
    An abstract class of an unnormalized conditional probability density
    function. This is intended to be used to represent a condiitonal model of
    the data    for goodness-of-fit testing. Specifically, the class specifies

    p(y|x) where the normalizer may not be known. That is, in fact, it
    specifies p(y,x) since p(y|x) = p(y,x)/p(x), and p(x) is the normalizer.
    The normalizer of the joint density is not assumed to be known.

    The KSSD and FSCD Stein-based tests only require grad_log(..). Subclasses
    can implement either log_den(..) or grad_log(..). If log_den(..) is
    implemented, grad_log(...) will be implemented automatically with
    torch.autograd functions. 
    """

    @abstractmethod
    def log_den(self, X, Y):
        """
        Evaluate log of the unnormalized density on the n points in (X, Y)
        i.e., log p(y_i, x_i) (up to the normalizer) for i = 1,..., n.
        Y, X are paired.

        X: n x dx Torch tensor
        Y: n x dy Torch tensor

        Return a one-dimensional Torch array of length n.
        The returned array A should be such that A[i] depends on only X[i] and Y[i].
        """
        expect_dx = self.dx()
        expect_dy = self.dy()
        if X.shape[1] != expect_dx:
            raise ValueError('X must have dimension dx={}. Found {}.'.format(expect_dx, X.shape[1]))
        if Y.shape[1] != expect_dy:
            raise ValueError('Y must have dimension dy={}. Found {}.'.format(expect_dy, Y.shape[1]))

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
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows. Found: X.shape[0] = {} and Y.shape[0] = {}'.format(X.shape[0], Y.shape[0]))
        
        # Default implementation with torch.autograd
        Y.requires_grad = True
        logprob = self.log_den(X, Y)
        # sum
        slogprob = torch.sum(logprob)
        Gs = torch.autograd.grad(slogprob, Y, retain_graph=True, only_inputs=True)
        G = Gs[0]

        n, dy = Y.shape
        assert G.shape[0] == n
        assert G.shape[1] == dy
        return G

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

class CDGaussianOLS(UnnormalizedCondDensity):
    """
    Implement p(y|x) = Normal(y - c - slope*x, variance) 
    which is the ordinary least-squares model with Gaussian noise N(0, variance).
    * Y is real valued.
    * X is dx dimensional. 
    """
    def __init__(self, slope, c, variance):
        """
        slope: the slope vector used in slope*x+c as the linear model. 
            One dimensional Torch tensor. Length of the slope vector
            determines the matching dimension of x.
        c: a bias (real value)
        variance: the variance of the noise
        """
        self.slope = slope.reshape(-1)
        self.c = c
        if variance <= 0:
            raise ValueError('variance must be positive. Was {}'.format(variance))
        self.variance = variance

    def log_den(self, X, Y):
        """
        log p(y_i, x_i) (the normalizer is optional) for i = 1,..., n.
        Y, X are paired.

        X: n x dx Torch tensor
        Y: n x dy Torch tensor

        Return a one-dimensional Torch array of length n.
        """
        super().log_den(X, Y)
        dx = self.dx()
        # https://pytorch.org/docs/stable/distributions.html#normal
        gauss = dists.Normal(0, self.variance)
        S = self.slope.reshape(dx, 1)
        Diff = Y - self.c - X.matmul(S)
        return gauss.log_prob(Diff)

    def get_condsource(self):
        """
        Return a CondSource that allows sampling from this density.
        """
        cs = cdat.CSGaussianOLS(self.slope, self.c, self.variance)
        return cs

    @abstractmethod
    def dy(self):
        """
        Return the dimension of Y.
        """
        return 1

    @abstractmethod
    def dx(self):
        """
        Return the dimension of X.
        """
        return self.slope.shape[0]


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

    