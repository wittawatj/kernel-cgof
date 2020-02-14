"""
Module containing data structures for representing datasets.
"""
__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
#import scipy.stats as stats
import kcgof
import kcgof.util as util
import torch

class CondData(object):
    """
    Class representing paired data {(y_i, x_i)}_{i=1}^n for conditional
    goodness-of-fit testing. The data are such that y_i is generated from a
    conditional distribution p(y|x_i).

    properties:
    X, Y: Pytorch tensor. X and Y are paired of the same sample size
        (.shape[0]). The dimensions (.shape[1]) are not necessarily the same.
    """

    def __init__(self, X, Y):
        """
        :param X: n x dx Pytorch tensor for dataset X
        :param Y: n x dy Pytorch tensor for dataset Y
        """
        self.X = X
        self.Y = Y

        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Sample size of the paired sample must be the same.')

        # if not np.all(np.isfinite(X)):
        #     print 'X:'
        #     print util.fullprint(X)
        #     raise ValueError('Not all elements in X are finite.')

        # if not np.all(np.isfinite(Y)):
        #     print 'Y:'
        #     print util.fullprint(Y)
        #     raise ValueError('Not all elements in Y are finite.')

    def dx(self):
        """Return the dimension of X."""
        dx = self.X.shape[1]
        return dx

    def dy(self):
        """Return the dimension of Y."""
        dy = self.Y.shape[1]
        return dy

    def n(self):
        return self.X.shape[0]

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    # def split_tr_te(self, tr_proportion=0.5, seed=820):
    #     """Split the dataset into training and test sets. Assume n is the same 
    #     for both X, Y. 
        
    #     Return (PairedData for tr, PairedData for te)"""
    #     X = self.X
    #     Y = self.Y
    #     nx, dx = X.shape
    #     ny, dy = Y.shape
    #     if nx != ny:
    #         raise ValueError('Require nx = ny')
    #     Itr, Ite = util.tr_te_indices(nx, tr_proportion, seed)
    #     label = '' if self.label is None else self.label
    #     tr_data = PairedData(X[Itr, :], Y[Itr, :], 'tr_' + label)
    #     te_data = PairedData(X[Ite, :], Y[Ite, :], 'te_' + label)
    #     return (tr_data, te_data)

    # def subsample(self, n, seed=87):
    #     """Subsample without replacement. Return a new PairedData """
    #     if n > self.X.shape[0] or n > self.Y.shape[0]:
    #         raise ValueError('n should not be larger than sizes of X, Y.')
    #     ind_x = util.subsample_ind( self.X.shape[0], n, seed )
    #     ind_y = util.subsample_ind( self.Y.shape[0], n, seed )
    #     return PairedData(self.X[ind_x, :], self.Y[ind_y, :], self.label)

# end PairedData class        

class CondSource(object):
    """
    Class representing a data source that allows one to generate data from a
    conditional distribution. This class basically implements a forward
    sampler of a conditional distribution p(y|x).
    No p(x) is implemented.

    Work with Pytorch tensors.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def cond_pair_sample(self, X, seed):
        """
        Return a Torch tensor Y such that Y.shape[0] = X.shape[0], and
        Y[i, :] ~ p(y | X[i, :]).
        The result should be deterministic given the seed value.
        """
        raise NotImplementedError()

    def __call__(self, X, seed):
        return self.cond_pair_sample(X, seed)

    @abstractmethod
    def dx(self):
        """Return the dimension of X"""
        raise NotImplementedError()
    
    @abstractmethod
    def dy(self):
        """Return the dimension of Y"""
        raise NotImplementedError()

# end class CondSource

class CSAdditiveNoiseRegression(CondSource):
    """
    CondSource for CDAdditiveNoiseRegression.
    Implement p(y|x) = f(x) + noise.
    The specified noise Z and its pdf g have to satisfy:
    Z + z_0 ~ g(Z - z_0) for any constant z_0.
    That is, g has to belong to the location family. Examples include the
    normal, Uniform, Laplace, etc.
    https://en.wikipedia.org/wiki/Location%E2%80%93scale_family
    """
    def __init__(self, f, noise, dx):
        """
        :param f: the mean function. A torch callable module.
            Return the same shape as input X.
        :param noise: an object with the same interface as a distribution in
            torch.distributions
                https://pytorch.org/docs/stable/distributions.html
            :param dx: dimension of x. A positive integer
        """
        self.f = f
        self.noise = noise
        self._dx = dx

    def cond_pair_sample(self, X, seed):
        if X.shape[1] != self._dx:
            raise ValueError('Input dimension in X (found {}) does not match dx ({}) as specified by the model.'.format(X.shape[1], self._dx))
        noise_dist = self.noise
        n = X.shape[0]
        # mean
        f = self.f
        fX = f(X)
        with util.TorchSeedContext(seed=seed):
            Y = noise_dist.sample((n, 1)) + fX
        return Y

    def dx(self):
        return self._dx

    def dy(self):
        return 1

# end CSAdditiveNoiseRegression

class CSGaussianOLS(CondSource):

    """
    A CondSource for sampling cdensity.CDGaussianOLS.

    p(y|x) = Normal(y - c - slope*x, variance) 
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
    
    def cond_pair_sample(self, X, seed):
        if X.shape[1] != self.slope.shape[0]:
            raise ValueError('The dimension of X must be the same as the dimension of slope. Slope dim: {}, X dim: {}'.format(self.slope.shape[0], X.shape[1]))
        n = X.shape[0]
        std = self.variance**0.5
        Mean = X.matmul(self.slope.reshape(self.dx(), 1)) + self.c
        with util.TorchSeedContext(seed=seed):
            sam = torch.randn(n,1)*std + Mean
        return sam

    def dx(self):
        return self.slope.shape[0]
    
    def dy(self):
        return 1
