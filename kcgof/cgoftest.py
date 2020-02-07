"""
Module containing statistical tests of goodness of fit of conditional density
models.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kcgof
import kcgof.util as util
import torch

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


class KSSDTest(CGofTest):
    """
    Conditional goodness-of-fit test with the Kernel-Smoothed Stein
    Discrepancy (KSSD).
    Test statistic is n*U-statistic.
    This test runs in O(n^2 d^2) time.

    H0: the joint sample follows p(y|x)
    H1: the joint sample does not follow p(y|x)

    p is specified to the constructor in the form of an UnnormalizedCondDensity.
    """

    def __init__(self, p, k, alpha=0.01, n_bootstrap=500, seed=11):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        bootstrapper: a function: (n) |-> numpy array of n weights 
            to be multiplied in the double sum of the test statistic for generating 
            bootstrap samples from the null distribution.
        alpha: significance level 
        n_bootstrap: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(KSSDTest, self).__init__(p, alpha)
        self.k = k
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def perform_test(self, dat, return_simulated_stats=False, return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        raise NotImplementedError()
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = n*np.mean(H)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            with util.NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                   W = self.bootstrapper(n)
                   # n * [ (1/n^2) * \sum_i \sum_j h(x_i, x_j) w_i w_j ]
                   boot_stat = W.dot(H.dot(old_div(W,float(n))))
                   # This is a bootstrap version of n*V_n
                   sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            pvalue = np.mean(sim_stats > test_stat)
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H
            
        return results


    def compute_stat(self, dat, return_ustat_gram=False):
        """
        Compute n times the U-statistic estimator of KSSD.

        return_ustat_gram: If true, then return the n x n matrix used to
            compute the statistic 
        """
        X = dat.data()
        n, d = X.shape
        k = self.k
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x n
        gram_glogp = grad_logp.dot(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        # V-statistic
        stat = n*np.mean(H)
        if return_ustat_gram:
            return stat, H
        else:
            return stat

        #print 't1: {0}'.format(t1)
        #print 't2: {0}'.format(t2)
        #print 't3: {0}'.format(t3)
        #print 't4: {0}'.format(t4)
