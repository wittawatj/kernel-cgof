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


    def _unsmoothed_ustat_kernel(self, X, Y):
        """
        Compute h_p((x,y), (x',y')) for (x,y) in X,Y.
        Return an n x n Torch tensor.
        """
        n, dy = Y.shape
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
        return h

    def compute_stat(self, X, Y, return_ustat_gram=False):
        """
        Compute n x the U-statistic estimator of KSSD.

        return_ustat_gram: If true, then return the n x n matrix used to
            compute the statistic 
        """
        n, dy = Y.shape
        k = self.k
        l = self.l
        h = self._unsmoothed_ustat_kernel(X, Y)
        # smoothing 
        K = k.eval(X, X)
        # TODO: Wittawat: K*h then sum, this is the same as forming a quadratic
        # form.
        H = K*h
        # U-statistic
        ustat = (torch.sum(H) - torch.sum(torch.diag(H)) )/(n*(n-1))
        stat = n*ustat
        if return_ustat_gram:
            return stat, H
        else:
            return stat

class FSCDPowerCriterion(object):
    """
    Construct a callable power criterion and witness functions associated
    with the FSCD test.
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
        self.kssdtest = KSSDTest(p, k, l)

    def eval_witness(self, at):
        """
        Evaluate the biased estimate of the witness function of KSSD/FSCD.

        at: Torch tensor of size m x dx specifying m locations to evaluate 
            the witness function. The witness function is evaluated at each
            point separately.

        Return: one-dimensional torch array of length m representing the
            values of the witness function evaluated at these locations.
        """
        # TODO: can be improved by vectorzing and avoiding the for loop. Later.
        return self._eval_witness_loop(at)

    def eval_power_criterion(self, at):
        """
        The power criterion is, by construction, a function of a set of test
        locations. So there are two modes of operation.

        at: If this is a Torch tensor of size J x dx, then evaluate the power
            criterion by treating the whole input tensor as one set of test
            locations. Return one scalar output.

            If this is a Torch tensor of size m x J x d, then interpret this
            as m sets of test locations to evaluate, and return m scalar
            outputs in a one-dimensional Torch array.
        """
        dim = len(at.shape)
        if dim == 2:
            return self._point_power_criterion(V=at)
        elif dim == 3:
            # TODO: try to improve the computation of this part. Not trivial
            # though.
            m, J, dx = at.shape
            pc_values = torch.zeros(m)
            for i in range(m):
                Vi = at[i]
                pc_values=[i] = self._point_power_criterion(V=Vi)
            return pc_values

        else:
            raise ValueError('at must be a 2d or a 3d tensor. Found at.shape = {}'.format(at.shape))

    def _point_power_criterion(self, V):
        """
        Evaluate the power criterion at the set of J locations in V.
        """
        kssdtest = self.kssdtest
        k = self.k

        h = kssdtest._unsmoothed_ustat_kernel(self.X, self.Y)
        n = h.shape[0]
        J, dx = V.shape

        # n x J
        Phi = k.eval(self.X, V)
        Kbar = Phi.matmul(Phi.T)/J
        # standard deviation under H1.
        hKbar = h*Kbar
        sigma_V = 2.0*torch.std(torch.mean(h*Kbar, 1))

        # compute biased FSCD = average of the witness values at the J
        # locations
        fscd_biased = torch.mean(hKbar)
        power_cri = fscd_biased/sigma_V
        return power_cri

    # def _point_h1_std(self, V):
    #     """
    #     Evaluate the standard deviation of the the distribution of FSCD under H1.
    #     Use V as the set of J test locations.
    #     """
    #     raise NotImplementedError()
    
    def _eval_witness_loop(self, at):
        """
        Same as eval_witness(.).
        This is the version with a for loop.
        Use eval_witness(.)
        """
        kssdtest = self.kssdtest
        # TODO: h can be cached if needed. But it may consume a lot of memory
        # (n x n)
        h = kssdtest._unsmoothed_ustat_kernel(self.X, self.Y)
        n = h.shape[0]
        # remove bias (diagonal)
        # h = h - torch.diagflat(torch.diag(h))
        m, dx = at.shape
        dy = self.Y.shape[1]
        k = self.k
        wit_values = torch.zeros(m)
        for i in range(m):
            loc_i = at[[i], :]
            # n x 1
            Phi = k.eval(self.X, loc_i)
            # print(h.matmul(Phi.reshape(-1)).dot(Phi.reshape(-1))/n**2)
            wit_values[i] = h.matmul(Phi.reshape(-1)).dot(Phi.reshape(-1))/(dy*n**2)
        return wit_values


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

