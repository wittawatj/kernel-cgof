"""
Module containing statistical tests of goodness of fit of conditional density
models.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kcgof
import kcgof.util as util
import kcgof.kernel as ker
import torch
import torch.distributions as dists
import typing

class CGofTest(object):
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
    def perform_test(self, X, Y) -> typing.Dict:
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

class KSSDWitness(object):
    """
    Construct a callable witness function associated with the KSSD test.
    The witness function is real-valued and is defined as 
    v |-> || G(v) ||^2
    where G is the RKHS-valued function such that its squared RKHS norm
    defines the KSSD statistic. The witness is supposed to be a zero function
    under H0. In practice, G has to be estimated from the data.
    """
    def __init__(self, p, k, l, X, Y):
        """
        p: an instance of UnnormalizedCondDensity
        k: a kernel.Kernel object representing a kernel on X
        l: a kernel.KCSTKernel object representing a kernel on Y
        X, Y: torch tensors representing the data for X and Y
        """
        self.p = p
        self.k = k
        self.l = l
        self.X = X
        self.Y = Y

    def __call__(self, at):
        """
        at: Torch tensor of size m x dx specifying m locations to evaluate 
            the witness function. The witness function is evaluated at each
            point separately.

        Return: one-dimensional torch array of length m representing the
            values of the witness function evaluated at these locations.
        """
        pass


class KSSDTest(CGofTest):
    """
    Conditional goodness-of-fit test with the Kernel-Smoothed Stein
    Discrepancy (KSSD).
    Test statistic is n*U-statistic.
    This test runs in O(n^2 d^2) time.

    H0: the joint sample follows p(y|x)
    H1: the joint sample does not follow p(y|x)

    p is specified to the constructor in the form of an
    UnnormalizedCondDensity.
    """

    def __init__(self, p, k, l, alpha=0.01, n_bootstrap=500, seed=11):
        """
        p: an instance of UnnormalizedCondDensity
        k: a kernel.Kernel object representing a kernel on X
        l: a kernel.KCSTKernel object representing a kernel on Y
        alpha: significance level 
        n_bootstrap: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        super(KSSDTest, self).__init__(p, alpha)
        self.k = k
        self.l = l
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def perform_test(self, X, Y, return_simulated_stats=False, return_ustat_gram=False):
        """
        X,Y: torch tensors. 
        return_simulated_stats: If True, also include the boostrapped
            statistics in the returned dictionary.
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_bootstrap = self.n_bootstrap
            n = X.shape[0]

            test_stat, H = self.compute_stat(X, Y, return_ustat_gram=True)
            # bootstrapping
            sim_stats = torch.zeros(n_bootstrap)
            mult_dist = dists.multinomial.Multinomial(total_count=n, probs=torch.ones(n)/n)
            with torch.no_grad():
                with util.TorchSeedContext(seed=self.seed):
                    for i in range(n_bootstrap):
                        W = mult_dist.sample()
                        Wt = (W-1.0)/n
                        # Bootstrapped statistic
                        boot_stat =  n * ( H.matmul(Wt).dot(Wt) - torch.diag(H).dot(Wt**2) )
                        sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            I = sim_stats > test_stat
            pvalue = torch.mean(I.type(torch.float)).item()
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 
            'test_stat': test_stat.item(),
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_bootstrap,
                 'time_secs': t.secs, 
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats.detach().numpy()
        if return_ustat_gram:
            results['H'] = H
            
        return results


    def compute_stat(self, X, Y, return_ustat_gram=False):
        """
        Compute n x the U-statistic estimator of KSSD.

        return_ustat_gram: If true, then return the n x n matrix used to
            compute the statistic 
        """
        n, dy = Y.shape
        k = self.k
        l = self.l
        # n x dy matrix of gradients
        grad_logp = self.p.grad_log(X, Y)
        # n x n
        gram_glogp = grad_logp.matmul(grad_logp.T)
        # n x n
        L = l.eval(Y, Y)

        B = torch.zeros((n, n))
        C = torch.zeros((n, n))
        for i in range(dy):
            grad_logp_i = grad_logp[:, i]
            B += l.gradX_Y(Y, Y, i)*grad_logp_i
            C += (l.gradY_X(Y, Y, i).T * grad_logp_i).T

        h = L*gram_glogp + B + C + l.gradXY_sum(Y, Y)
        # smoothing 
        K = k.eval(X, X)
        H = K*h
        # U-statistic
        ustat = (torch.sum(H) - torch.sum(torch.diag(H)) )/(n*(n-1))
        stat = n*ustat
        if return_ustat_gram:
            return stat, H
        else:
            return stat

class FSCDTest(KSSDTest):
    """
    Conditional goodness-of-fit test with the Finite Set Conditional
    Discrepancy (FSCD).

    Test statistic is n*U-statistic.

    H0: the joint sample follows p(y|x)
    H1: the joint sample does not follow p(y|x)

    p is specified to the constructor in the form of an
    UnnormalizedCondDensity.
    """
    def __init__(self, p, k, l, V, alpha=0.01, n_bootstrap=500, seed=12):
        """
        p: an instance of UnnormalizedCondDensity
        k: a kernel.Kernel object representing a kernel on X
        l: a kernel.KCSTKernel object representing a kernel on Y
        V: torch array of size J x dx representing the J test locations in
            the domain of X
        alpha: significance level 
        n_bootstrap: The number of times to simulate from the null distribution
            by bootstrapping. Must be a positive integer.
        """
        # form a finite-dimensional kernel defined with the test locations
        kbar = ker.PTKTestLocations(k, V)
        super(FSCDTest, self).__init__(p, kbar, l, alpha=alpha,
            n_bootstrap=n_bootstrap, seed=seed)
        self.V = V

