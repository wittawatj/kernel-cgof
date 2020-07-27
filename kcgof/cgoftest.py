"""
Module containing statistical tests of goodness of fit of conditional density
models.
"""

__author__ = 'wittawat'

from abc import ABCMeta, abstractmethod
import kcgof
import kcgof.util as util
import kcgof.kernel as ker
import kcgof.cdensity as cd
import kcgof.cdata as cdat
import torch
import torch.distributions as dists
import torch.optim as optim
import typing
from scipy.integrate import quad
import numpy as np
import logging


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

        All values in the rutned dictionary should be scalar or numpy arrays
        if possible (avoid torch tensors).
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, X, Y):
        """
        Compute the test statistic. 
        Return a scalar value.
        """
        raise NotImplementedError()


class KCSDTest(CGofTest):
    """
    Conditional goodness-of-fit test with the Kernel Conditional Stein
    Discrepancy (KCSD).
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
        super(KCSDTest, self).__init__(p, alpha)
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
        Compute n x the U-statistic estimator of KCSD.

        return_ustat_gram: If true, then return the n x n matrix used to
            compute the statistic 
        """
        n, dy = Y.shape
        k = self.k
        l = self.l
        h = self._unsmoothed_ustat_kernel(X, Y)
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

class KCSDPowerCriterion(object):
    """
    Implement the power criterion of the KCSD test for parameter tuning of the test.
    Related: see also FSCDPowerCriterion.
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
        self.kcsdtest = KCSDTest(p, k, l)
    
    def optimize_params(self, params, lr, constraint_f=None, reg=1e-4,
        max_iter=500):
        """
        Optimize parameters in the list params by maximizing the power
        criterion of the KCSD test. This method modifies the state of this
        object (specifically, parameters in k, l).

        - params:  a list of torch.Tensor s or dict s.
        Specifies what Tensors should be optimized. Will be fed to an
        optimizer in torch.optim. All parameters in params must be part of
        (p, k, l). 

        - constraint_f: callable object (params) |-> None that modifies
        all the parameters to be optimized in-place to satisfy the
        constraints (if any).

        - reg: regularizer of the power criterion

        - lr: overall learning rate. Lr of each parameter can be specified
        separately as well. https://pytorch.org/docs/stable/optim.html

        - max_iter: maximum number of gradient updates

        Return a torch array of recorded function values
        """
        if params is None:
            params = []
        if constraint_f is None:
            constraint_f = lambda *args, **kwargs: None
        # optimizer
        all_params = params
        for pa in all_params:
            pa.requires_grad = True
        optimizer = optim.Adam(all_params, lr=lr)

        # record
        objs = torch.zeros(max_iter)
        for t in range(max_iter):
            optimizer.zero_grad()
            # minimize the *negative* of power criterion
            obj = -self._point_power_criterion(reg=reg)
            obj.backward()
            optimizer.step()
            # constraint satisfaction
            constraint_f(params)
            # Flip the sign back
            objs[t] = -obj.detach()
        return objs

    def _point_power_criterion(self, reg=1e-5):
        """
        Evaluate the regularized power criterion of KCSD test using the
        specified kernels and data.
        The objective is mean_under_H1 / (reg + standard deviation under H1)

        reg: a non-negative scalar specifying the regularization parameter
        """
        kcsdtest = self.kcsdtest
        k = self.k

        h = kcsdtest._unsmoothed_ustat_kernel(self.X, self.Y)
        n = h.shape[0]
        K = k.eval(self.X, self.X)
        # standard deviation under H1.
        hK = h*K
        sigma_h1 = 2.0*torch.std(torch.mean(hK, 1))

        # compute biased KCSD 
        kcsd_biased = torch.mean(hK)
        power_cri = kcsd_biased/(sigma_h1 + reg)
        return power_cri

class FSCDPowerCriterion(object):
    """
    Construct a callable power criterion and witness functions associated
    with the FSCD test.
    The witness function is real-valued and is defined as 
    v |-> || G(v) ||^2
    where G is the RKHS-valued function such that its squared RKHS norm
    defines the KCSD statistic. The witness is supposed to be a zero function
    under H0. In practice, G has to be estimated from the data.

    High power criterion indicates a poor fit of the model on the data.
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
        self.kcsdtest = KCSDTest(p, k, l)
    
    def eval_witness(self, at):
        """
        Evaluate the biased estimate of the witness function of KCSD/FSCD.

        at: Torch tensor of size m x dx specifying m locations to evaluate 
            the witness function. The witness function is evaluated at each
            point separately.

        Return: one-dimensional torch array of length m representing the
            values of the witness function evaluated at these locations.
        """
        # TODO: can be improved by vectorzing and avoiding the for loop. Later.
        return self._eval_witness_loop(at)

    def eval_power_criterion(self, at, reg=1e-5):
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
            return self._point_power_criterion(V=at, reg=reg)
        elif dim == 3:
            # TODO: try to improve the computation of this part. Not trivial
            # though.
            m, J, dx = at.shape
            pc_values = torch.zeros(m)
            for i in range(m):
                Vi = at[i]
                # print(Vi)
                # detaching saves a lot of memory
                pc_values[i] = self._point_power_criterion(V=Vi, reg=reg).detach()
            return pc_values
        else:
            raise ValueError('at must be a 2d or a 3d tensor. Found at.shape = {}'.format(at.shape))

    def optimize_params(self, params, V, lr, constraint_f=None, reg=1e-4, max_iter=500):
        """
        Optimize parameters in the list params by maximizing the power
        criterion of the FSCD test. This method modifies the state of this
        object (specifically, parameters in k, l).

        - params:  a list of torch.Tensor s or dict s.
        Specifies what Tensors should be optimized. Will be fed to an
        optimizer in torch.optim. All parameters in params must be part of
        (p, k, l). 

        - V: J x dx test locations

        - constraint_f: callable object (params, V) |-> None that modifies
        all the parameters to be optimized in-place to satisfy the
        constraints (if any).

        - reg: regularizer of the power criterion

        - lr: overall learning rate. Lr of each parameter can be specified
        separately as well. https://pytorch.org/docs/stable/optim.html

        - max_iter: maximum number of gradient updates

        Return a torch array of recorded function values
        """
        if params is None:
            params = []
        if constraint_f is None:
            constraint_f = lambda *args, **kwargs: None
        # optimizer
        all_params = params + [V]
        for pa in all_params:
            pa.requires_grad = True
        optimizer = optim.Adam(all_params, lr=lr)

        # record
        objs = torch.zeros(max_iter)
        for t in range(max_iter):
            optimizer.zero_grad()
            # minimize the *negative* of power criterion
            obj = -self._point_power_criterion(V, reg)
            obj.backward()
            optimizer.step()
            # constraint satisfaction
            constraint_f(params, V)
            # Flip the sign back
            objs[t] = -obj.detach()
        return objs

    def _point_power_criterion(self, V, reg=1e-5):
        """
        Evaluate the regularized power criterion at the set of J locations in
        V. The objective is mean_under_H1 / (reg + standard deviation under H1)

        reg: a non-negative scalar specifying the regularization parameter
        """
        kcsdtest = self.kcsdtest
        k = self.k

        h = kcsdtest._unsmoothed_ustat_kernel(self.X, self.Y)
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
        power_cri = fscd_biased/(sigma_V + reg)
        return power_cri

    # def _point_h1_std(self, V):
    #     """
    #     Evaluate the standard deviation of the the distribution of FSCD under H1.
    #     Use V as the set of J test locations.
    #     """
    #     kcsdtest = self.kcsdtest
    #     k = self.k

    #     h = kcsdtest._unsmoothed_ustat_kernel(self.X, self.Y)
    #     n = h.shape[0]
    #     J, dx = V.shape

    #     # n x J
    #     Phi = k.eval(self.X, V)
    #     Kbar = Phi.matmul(Phi.T)/J
    #     # standard deviation under H1.
    #     hKbar = h*Kbar
    #     sigma_V = 2.0*torch.std(torch.mean(h*Kbar, 1))
    #     return sigma_V
    
    def _eval_witness_loop(self, at):
        """
        Same as eval_witness(.).
        This is the version with a for loop.
        Use eval_witness(.)
        """
        kcsdtest = self.kcsdtest
        # TODO: h can be cached if needed. But it may consume a lot of memory
        # (n x n)
        h = kcsdtest._unsmoothed_ustat_kernel(self.X, self.Y)
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


class FSCDTest(KCSDTest):
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


class ZhengKLTest(CGofTest):
    """
    An implementation of 
    "Zheng 2000, A CONSISTENT TEST OF CONDITIONAL PARAMETRIC DISTRIBUTIONS", 
    which uses the first order approximation of KL divergence as the decision
    criterion. 
    Currently this class only supports conditional density with output 
    dimension 1. 
    The model paramter is assumed to be fixed at the best one (no estimator). 
    Args: 
        p: an instance of UnnormalizedDensity
        alpha: significance level
        kx: smoothing kernel function for covariates. Default is Zheng's kernel.
        ky: smoothing kernel function for output variables. Default is Zheng's kernel.
    """

    def __init__(self, p, alpha, kx=None, ky=None, rate=0.2):
        super(ZhengKLTest, self).__init__(p, alpha)
        if p.dy() != 1:
            raise ValueError(('this test can be used only '
                              'for 1-d y'))
        if not hasattr(p, 'log_normalized_den'):
            raise ValueError('the density needs to be normalized')
        self.kx = kx if kx is not None else ZhengKLTest.K1
        self.ky = ky if ky is not None else ZhengKLTest.K2
        self.rate = rate

    def _integrand(self, y, y0, x, h):
        y_ = torch.from_numpy(np.array(y)).type(torch.float).view(1, -1)
        y0_ = torch.from_numpy(np.array(y0)).type(torch.float).view(1, -1)
        x_ = torch.from_numpy(np.array(x)).type(torch.float).view(1, -1)
        val = self.ky((y0_-y_)/h, h) * torch.exp(self.p.log_normalized_den(x_, y_))
        return val.numpy()

    def integrate(self, y0, x, h, lb=-np.inf, ub=np.inf):
        inted = quad(self._integrand, lb, ub, args=(y0, x, h), epsabs=1.49e-3, limit=10)[0]
        return inted

    def compute_stat(self, X, Y, h=None): 
        """
        Compute the test static. 
        h: optinal kernel width param
        """

        def integrate_gaussleg(y0, x, h, lb=-10, ub=10, n_nodes=10):
            """
            Numerically integrate the integral in the statistic of Zheng 2000
            with Gauss-Legendre.

            n_nodes: number of nodes used to approximate the integral
            """
            # TODO: What should be the value of n_nodes?
            import numpy
            from numpy.polynomial import legendre

            f_int = lambda yy: self._integrand(yy, y0, x, h)
            YY, W = legendre.leggauss(n_nodes)

            #https://en.wikipedia.org/wiki/Gaussian_quadrature
            f_arg = (ub-lb)/2.0*YY + (ub+lb)/2.0 
            f_arg = f_arg.reshape(-1, 1)
            f_eval_values = np.zeros(n_nodes)
            for i in range(n_nodes):
                f_eval_values[i] = f_int(f_arg[i])
            
            # f_eval_values = f_int(f_arg)
            gaussleg_int = 0.5*(ub-lb)*W.dot( f_eval_values ) 
            return gaussleg_int

        def vec_integrate(K1, Y, X, h):
            """
            K1: n x n_
            K1 can contain zeros. Do not do numerical integration in the cell
                [i,j] where K1[i,j] = 0 = 0
            """
            int_results = np.empty([Y.shape[0], X.shape[0]])
            # TODO: What should the integral width be? Depends on h?
            integral_width = 1.0
            n = Y.shape[0]
            for i in range(n):
                for j in range(i, n):
                    if torch.abs(K1[i, j]) <= 1e-7: # 0
                        int_results[i,j]= 0.0
                        int_results[j, i] = 0.0

                    else:
                        # Previously we used integrate(..) which uses quad(..)
                        # print(X[j])
                        int_quad = self.integrate(Y[i], X[j], h)
                        # Add the following line just to print integrated values
                        # print('quad integrate: ', int_quad)
                        # int_gaussleg  = integrate_gaussleg(
                        #     Y[i], X[j], h, 
                        #     lb=Y[i].item()-integral_width, ub=Y[i].item()+integral_width)
                        # print('Gauss-Legendre: {}'.format(int_gaussleg))
                        # print()

                        int_results[i, j] = int_quad
                        int_results[j, i] = int_results[i, j]
            return int_results

        n, dx = X.shape
        dy = Y.shape[1]
        if h is None:
            h = n**((self.rate-1.)/(dx+dy))

        # K1: n x n
        K1 = self.kx((X.unsqueeze(1)-X)/h)
        # print(K1)
        K2 = self.ky((Y.unsqueeze(1)-Y)/h, h)

        integrated = torch.from_numpy(vec_integrate(K1, Y, X, h))
        # vec_integrate_ = np.vectorize(integrate, signature='(n),(m),()->()')
        # integrated = torch.from_numpy(vec_integrate_(Y.reshape([n, dy]), X, h))

        # K contains values of the numerator in Eq 2.12 of Zheng 2000. n x n
        K = K1 * (K2 - integrated)
        log_den = self.p.log_normalized_den(X, Y)
        K /= torch.exp(log_den)

        var = K1**2
        var = 2. * (torch.sum(var)-torch.sum(torch.diag(var)))
        var = var / h**(dx) / (n*(n-1))

        stat = (torch.sum(K) - torch.sum(torch.diag(K))) / (n*(n-1))
        # Statistic = Eq. 2.13 in Zheng 2000
        stat *= n * h**(-(dx+dy)/2) / var**0.5
        return stat

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
        with util.ContextTimer() as t:
            alpha = self.alpha
            stat = self.compute_stat(X, Y)
            pvalue = (1 - dists.Normal(0, 1).cdf(stat)).item()

        results = {'alpha': self.alpha, 'pvalue': pvalue,
                   'test_stat': stat.item(),
                   'h0_rejected': pvalue < alpha, 'time_secs': t.secs,
                   }
        return results

    @staticmethod
    def K1(X):
        """
        Kernel function for explanation variables used in Zheng's paper.
        Dimension-wise product of Epanechnikov kernel.

        X: Torch tensor of size n x dx
        Return: Evaluated kernel value of size n
        """
        K = torch.zeros(X.shape)
        idx = (torch.abs(X) <= 1.)
        K[idx] = 0.75 * (1 - X[idx]**2)
        return torch.prod(K, dim=-1)

    @staticmethod
    def K2(Y, h):
        """
        Kernel function for dependent variables used in Zheng's paper. 
        Y: Torch tensor of size n x dy
        Return: kernel evaluated at Y of size n
        """
        K = torch.zeros(Y.shape)
        weight = 1 - torch.exp(-2./h)
        pos_idx = ((Y>=0) & (Y<=1./h)).prod(dim=-1).bool()
        K[pos_idx] = 2.*torch.exp(-2.*Y[pos_idx]) / weight
        neg_idx = ((Y<0) & (Y>=-1./h)).prod(dim=-1).bool()
        K[neg_idx] = 2.*torch.exp(-2.*(Y[neg_idx]+1./h)) / weight
        return torch.prod(K, dim=-1)


class ZhengKLTestMC(ZhengKLTest):
    """ 
    Zheng 2000 test without the numerical integration. See ZhengKLTest for
    another version with numerical integration. In this version, samples are
    drawn from the conditional model instead. Require that the specified
    model has a get_condsource(..) implemented.

    This Monte Carlo version is done to speed up.
    """
    def __init__(self, p, alpha, n_mc=2000, kx=None, ky=None, rate=0.2, verbose=False):
        """
        n_mc: number of samples to use for the Monte Carlo integration
        verbose: if true, print debugging information.
        """
        super(ZhengKLTestMC, self).__init__(p, alpha, kx, ky, rate)
        self.n_mc = n_mc
        self.verbose = verbose
        if p.dy() != 1:
            raise ValueError(('this test can be used only '
                              'for 1-d y'))
        if p.get_condsource() is None:
            raise ValueError('This test requires a way to sample from the model. The model p needs to implement get_condsource().')

    def compute_stat(self, X, Y, h=None): 
        """
        Compute the test static. 
        h: optinal kernel width param
        """
        n, dx = X.shape
        dy = Y.shape[1]
        Z = torch.sigmoid(Y)
        if h is None:
           # h = n**((self.rate-1.)/(dx+dy))
           h = torch.std(X, dim=0).mean() * n**((self.rate-1.)/(dx+dy))
        p = self.p
        # requires a CondSource
        cs = p.get_condsource()

        # K1: n x n
        K1 = self.kx((X.unsqueeze(1)-X)/h)
        # print(K2)
        K2 = self.ky((Z.unsqueeze(1)-Z)/h, h)

        def vec_montecarlo(K1, Y, X, h, n_sample):
            """
            K1: n x n_
            K1 can contain zeros. Do not do numerical integration in the cell
                [i,j] where K1[i,j] =  0

            n_sample: number of samples to draw from the conditional model 
                to do Monte Carlo integration.
            """
            int_results = np.empty([Y.shape[0], X.shape[0]])
            # TODO: What should the integral width be? Depends on h?
            n = Y.shape[0]
            # Z = ZhengKLTest.logistic(Y)
            Z = torch.sigmoid(Y)
            for i in range(n):
                for j in range(i, n):
                    if torch.abs(K1[i, j]) <= 1e-7: # 0
                        int_results[i, j] = 0.0
                        int_results[j, i] = 0.0

                    else:
                        # Monte Carlo integration
                        # Sample from model p(y|x_j)
                        XXj = X[j].reshape(1, dx).repeat(n_sample, 1)
                        # sample
                        YYj = cs(XXj, seed=587)
                        ZZj = torch.sigmoid(YYj)
                        KZZj = self.ky((Z[i] - ZZj)/h, h)
                        int_mc = torch.mean(KZZj)

                        if self.verbose:
                            print('MC integrate: {}'.format(int_mc))
                            # Add the following line just to print quad (expensive) integrated values
                            int_quad = self.integrate(Y[i], X[j], h)
                            print('quad integrate: ', int_quad)
                            print()

                        int_results[i, j] = int_mc
                        int_results[j, i] = int_results[i, j]
            return int_results

        integrated = torch.from_numpy(vec_montecarlo(K1, Y, X, h, self.n_mc))
        # vec_integrate_ = np.vectorize(integrate, signature='(n),(m),()->()')
        # integrated = torch.from_numpy(vec_integrate_(Y.reshape([n, dy]), X, h))

        # K contains values of the numerator in Eq 2.12 of Zheng 2000. n x n
        K = K1 * (K2 - integrated)
        log_den = self.p.log_normalized_den(X, Y)
        K /= torch.exp(log_den)*(1./(1.-Z)+1./Z)

        var = K1**2
        var = 2. * (torch.sum(var)-torch.sum(torch.diag(var)))
        var = var / (n*(n-1))

        stat = (torch.sum(K) - torch.sum(torch.diag(K))) / (n*(n-1))
        # Statistic = Eq. 2.13 in Zheng 2000
        stat *= n * h**((dx-dy)/2.0) / var**0.5
        return stat


class ZhengKLTestGaussHerm(ZhengKLTest):
    """
    An implementation of 
    "Zheng 2000, A CONSISTENT TEST OF CONDITIONAL PARAMETRIC DISTRIBUTIONS", 
    which uses the first order approximation of KL divergence as the decision
    criterion. 
    Currently this class only supports conditional density with output 
    dimension 1. 
    This is a class specialised for OLS model with Gaussian noise. 

    The model paramter is assumed to be fixed at the best one (no estimator). 
    Args: 
        p: an instance of UnnormalizedDensity
        alpha: significance level
        kx: smoothing kernel function for covariates. Default is Zheng's kernel.
        ky: smoothing kernel function for output variables. Default is Zheng's kernel.
    """

    def __init__(self, p, alpha, kx=None, ky=None, rate=0.2):
        super(ZhengKLTestGaussHerm, self).__init__(p, alpha, kx, ky, rate)
        if type(p) is not cd.CDGaussianOLS:
            raise ValueError('This method is only for Gaussian CD.')

    def _integrand_wo_gaussian(self, y, y0, x, h):
        from math import pi
        slope = self.p.slope
        c = self.p.c
        mean = x @ slope + c
        std = self.p.variance**0.5
        y_ = torch.from_numpy(np.array(y)).type(torch.float).view(1, -1)
        y0_ = torch.from_numpy(np.array(y0)).type(torch.float).view(1, -1)
        x_ = torch.from_numpy(np.array(x)).type(torch.float).view(1, -1)
        val = self.ky((y0_-(2**0.5*std*y_+mean))/h, h) / (pi**0.5)
        return val.numpy()

    def integrate(self, y0, x, h, lb=-np.inf, ub=np.inf):
        inted = quad(self._integrand, lb, ub, args=(y0, x, h), epsabs=1.49e-3, limit=10)[0]
        return inted

    def compute_stat(self, X, Y, h=None): 
        """
        Compute the test static. 
        h: optinal kernel width param
        """

        def integrate_gaussherm(y0, x, h, deg=5):
            """
            Numerically integrate the integral in the statistic of Zheng 2000
            with Gauss-Hermitite quadrature.

            deg: degree of polynomials
            """
            import numpy
            from numpy.polynomial.hermite import hermgauss
            points, weights = hermgauss(deg)
            n = len(weights)
            vec_evals = np.empty(n)
            for i in range(n):
                vec_evals[i] = self._integrand_wo_gaussian(points[i], y0,
                                                           x, h)
            integrated = weights.dot(vec_evals)
            return integrated

        def vec_integrate(K1, Y, X, h):
            """
            K1: n x n_
            K1 can contain zeros. Do not do numerical integration in the cell
                [i,j] where K1[i,j] = 0 = 0
            """
            int_results = np.empty([Y.shape[0], X.shape[0]])
            # TODO: What should the integral width be? Depends on h?
            integral_width = 1.0
            n = Y.shape[0]
            for i in range(n):
                for j in range(i, n):
                    if torch.abs(K1[i, j]) <= 1e-7: # 0
                        int_results[i,j]= 0.0
                        int_results[j, i] = 0.0

                    else:
                        # Previously we used integrate(..) which uses quad(..)
                        # print(X[j])
                        #int_quad = self.integrate(Y[i], X[j], h)
                        # Add the following line just to print integrated values
                        #print('quad integrate: ', int_quad)
                        # We use Gaussian Hermite quadrature
                        int_gaussherm = integrate_gaussherm(Y[i], X[j], h)
                        # print('Gauss-Herm: {}'.format(int_gaussherm))
                        # print()

                        # int_results[i, j] = int_quad
                        int_results[i, j] = int_gaussherm
                        int_results[j, i] = int_results[i, j]
            return int_results

        n, dx = X.shape
        dy = Y.shape[1]
        if h is None:
           h = n**((self.rate-1.)/(dx+dy))

        # K1: n x n
        K1 = self.kx((X.unsqueeze(1)-X)/h)
        # print(K1)
        K2 = self.ky((Y.unsqueeze(1)-Y)/h, h)

        integrated = torch.from_numpy(vec_integrate(K1, Y, X, h))
        # vec_integrate_ = np.vectorize(integrate, signature='(n),(m),()->()')
        # integrated = torch.from_numpy(vec_integrate_(Y.reshape([n, dy]), X, h))

        # K contains values of the numerator in Eq 2.12 of Zheng 2000. n x n
        K = K1 * (K2 - integrated)
        log_den = self.p.log_normalized_den(X, Y)
        K /= torch.exp(log_den)

        var = K1**2
        var = 2. * (torch.sum(var)-torch.sum(torch.diag(var)))
        var = var / h**(dx) / (n*(n-1))

        stat = (torch.sum(K) - torch.sum(torch.diag(K))) / (n*(n-1))
        # Statistic = Eq. 2.13 in Zheng 2000
        stat *= n * h**(-(dx+dy)/2) / var**0.5
        return stat

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
        with util.ContextTimer() as t:
            alpha = self.alpha
            stat = self.compute_stat(X, Y)
            pvalue = (1 - dists.Normal(0, 1).cdf(stat)).item()

        results = {'alpha': self.alpha, 'pvalue': pvalue,
                   'test_stat': stat.item(),
                   'h0_rejected': pvalue < alpha, 'time_secs': t.secs,
                   }
        return results

    
class MMDTest(CGofTest):
    """
    A MMD test for a goodness-of-fit test for conditional density models. 

    Args: 
        p: an instance of UnnormalizedCondDensity
        k: a kernel.Kernel object representing a kernel on X
        l: a kernel.KCSTKernel object representing a kernel on Y
        n_permute: number of times to permute the samples to simulate from the 
            null distribution (permutation test)
        alpha (float): significance level 
        seed: random seed
    """

    def __init__(self, p, k, l, n_permute=400, alpha=0.01, seed=11):
        # logging.warning(('This test does not accept Pytorch '
        #                  'kernels starting with prefix PT'))

        import freqopttest.tst as tst
        super(MMDTest, self).__init__(p, alpha)
        self.p = p
        self.k = k
        self.l = l
        self.ds_p = self.p.get_condsource()
        if self.ds_p is None:
            raise ValueError('The test requires that p can be sampled. Must implement p.get_condsource().')
        self.alpha = alpha
        self.seed = seed
        self.n_permute = n_permute
        kprod = ker.KTwoProduct(k, l, p.dx(), p.dy())
        self.mmdtest = tst.QuadMMDTest(kprod, n_permute, alpha=alpha)

    def compute_stat(self, X, Y):
        """
        X: Torch tensor of size n x dx
        Y: Torch tensor of size n x dy
        
        Return a test statistic
        """
        import freqopttest.data as fdata
        seed = self.seed
        ds_p = self.ds_p
        mmdtest = self.mmdtest

        # Draw sample from p
        Y_ = ds_p.cond_pair_sample(X, seed=seed+13)
        real_data = torch.cat([X, Y], dim=1).numpy()
        model_data = torch.cat([X, Y_], dim=1).numpy()
        # Make a two-sample test data
        tst_data = fdata.TSTData(real_data, model_data)
        stat = mmdtest.compute_stat(tst_data)
        return stat

    def perform_test(self, X, Y):
        import freqopttest.data as fdata
        ds_p = self.ds_p
        mmdtest = self.mmdtest
        seed = self.seed

        with util.ContextTimer() as t:
            # Draw sample from p
            Y_ = ds_p.cond_pair_sample(X, seed=seed+13)
            real_data = torch.cat([X, Y], dim=1).numpy()
            model_data = torch.cat([X, Y_], dim=1).numpy()
 
            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = fdata.TSTData(real_data, model_data)
            # Test 
            results = mmdtest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results

class MMDSplitTest(CGofTest):
    """
    Same as the MMDTest but split the data (X,Y) into two parts:
    X1, X2 and Y1, Y2.

    Sample Y2' from the model with p(y|X2).
    Test the difference between (X1, Y1) and (X2, Y2') with the MMD.
    The splitting step is to ensure that the two sets of samples are independent.
    Note that we lose some real samples in the process.
    """

    def __init__(self, p, k, l, n_permute=400, alpha=0.01, seed=101):
        # logging.warning(('This test does not accept Pytorch '
        #                  'kernels starting with prefix PT'))
        import freqopttest.tst as tst
        super(MMDSplitTest, self).__init__(p, alpha)

        self.p = p
        self.k = k
        self.l = l
        self.ds_p = self.p.get_condsource()
        if self.ds_p is None:
            raise ValueError('The test requires that p can be sampled. Must implement p.get_condsource().')
        self.alpha = alpha
        self.seed = seed
        self.n_permute = n_permute

        kprod = ker.KTwoProduct(k, l, p.dx(), p.dy())
        self.mmdtest = tst.QuadMMDTest(kprod, n_permute, alpha=alpha)

    @staticmethod
    def _split_half(X, Y, seed=28355):
        n = X.shape[0]
        if n%2 != 0:
            # odd
            X = X[:-1]
            Y = Y[:-1]

        # split into two halves of equal sizes
        dat1, dat2 = cdat.CondData(X, Y).split_tr_te(tr_proportion=0.5, seed=seed)
        X1, Y1 = dat1.xy()
        X2, Y2 = dat2.xy()
        return X1, Y1, X2, Y2

    def compute_stat(self, X, Y):
        """
        X: Torch tensor of size n x dx
        Y: Torch tensor of size n x dy
        
        Return a test statistic
        """

        import freqopttest.data as fdata
        seed = self.seed
        ds_p = self.ds_p
        mmdtest = self.mmdtest
        # split the data
        X1, Y1, X2, Y2 = MMDSplitTest._split_half(X, Y, seed=self.seed+330)

        # Draw sample from p
        Y2_ = ds_p.cond_pair_sample(X2, seed=seed+13)
        real_data = torch.cat([X1, Y1], dim=1).numpy()
        model_data = torch.cat([X2, Y2_], dim=1).numpy()
        # Make a two-sample test data
        tst_data = fdata.TSTData(real_data, model_data)
        stat = mmdtest.compute_stat(tst_data)
        return stat

    def perform_test(self, X, Y):

        import freqopttest.data as fdata
        ds_p = self.ds_p
        mmdtest = self.mmdtest
        seed = self.seed

        with util.ContextTimer() as t:
            # split the data
            X1, Y1, X2, Y2 = MMDSplitTest._split_half(X, Y, seed=self.seed+330)

            # Draw sample from p
            Y2_ = ds_p.cond_pair_sample(X2, seed=seed+13)
            real_data = torch.cat([X1, Y1], dim=1).numpy()
            model_data = torch.cat([X2, Y2_], dim=1).numpy()
 
            # Run the two-sample test on p_sample and dat
            # Make a two-sample test data
            tst_data = fdata.TSTData(real_data, model_data)
            # Test 
            results = mmdtest.perform_test(tst_data)

        results['time_secs'] = t.secs
        return results


class CramerVonMisesTest(CGofTest): 

    """
    Misspecification Testing in a Class of Conditional Distributional Models
    """

    def __init__(self, p, n_bootstrap=100, alpha=0.01, seed=11):
        if type(p) is not cd.CDGaussianOLS:
                raise ValueError('This method is only for Gaussian CD.')
        self.p = p 
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.seed = seed 
   
    @staticmethod
    def pairwise_comparison(X, X_):  
        """
        X: n x d torch tensor 
        X_: n x d torch tensor 
        Return: a torch tensor of size n x n whose 
        (i, j) element is indicator function of X_i <= (X_)_j
        """
        return (1.*(X <= X_.unsqueeze(1)).prod(dim=-1)).T

    @staticmethod
    def Hn(X, Y, X_, Y_):
        """
        X: n x d torch tensor 
        Y: n x d torch tensor 
        X: n x d torch tensor 
        Return: torch tensor of size n whose ith element is the empirical joint CDF 
        constructed from X, Y and evaluated at X_ and Y_
        """
        n = X.shape[0]
        Xpart = CramerVonMisesTest.pairwise_comparison(X, X_)
        Ypart = CramerVonMisesTest.pairwise_comparison(Y, Y_)
        return torch.mean(Xpart * Ypart, dim=0)

    def Hn0(self, X, Y, X_, Y_):
        n = X.shape[0]
        p = self.p 
        mean = X @ p.slope + p.c
        std = self.p.variance**0.5
        Hn0 = torch.zeros(n, n)

        norms = [dists.Normal(mean[i], std * torch.eye(p.dy()))
                 for i in range(n)]
        for j in range(n):
            norm = norms[j]
            Hn0[:, j] = norm.cdf(Y_).squeeze()
        Hn0 *= CramerVonMisesTest.pairwise_comparison(X, X_).T
        Hn0 = Hn0.mean(dim=1)
        return Hn0

    def compute_stat(self, X, Y): 
        n = X.shape[0]
        Hn = (CramerVonMisesTest.Hn(X, Y, X, Y))
        Hn0 = self.Hn0(X, Y, X, Y)
        return torch.sum((Hn - Hn0)**2)

    def perform_test(self, X, Y):
        with util.ContextTimer() as t:
            alpha = self.alpha
            n_bootstrap = self.n_bootstrap
            n = X.shape[0]
            ds = self.p.get_condsource()

            test_stat = self.compute_stat(X, Y)
            # bootstrapping
            sim_stats = torch.zeros(n_bootstrap)
            with torch.no_grad():
                with util.TorchSeedContext(seed=self.seed):
                    for i in range(n_bootstrap):
                        idx = torch.randint(0, n, [n])
                        X_ = X[idx]
                        Y_ = ds.cond_pair_sample(X_, self.seed+i)
                        # Bootstrapped statistic
                        Hnb = CramerVonMisesTest.Hn(X_, Y_, X, Y)
                        Hn0b = self.Hn0(X_, Y_, X, Y)
                        boot_stat = torch.sum((Hnb - Hn0b)**2)
                        sim_stats[i] = boot_stat
 
            # approximate p-value with the permutations 
            I = sim_stats > test_stat
            pvalue = torch.mean(I.type(torch.float)).item()
 
        results = {'alpha': self.alpha, 'pvalue': pvalue, 
                   'test_stat': test_stat.item(),
                   'h0_rejected': pvalue < alpha, 'n_simulate': n_bootstrap,
                   'time_secs': t.secs,
                   }
        return results


class ZhengCDFTest(CGofTest):
    """
    Zheng's test with a statistic based on a difference 
    between empirical and model CDFs,
    implementing "Testing parametric conditional distributions using
    the nonparametric smoothing method". 

    Currently, this class only supports
        - CDGaussianOLS
        - CDGaussianHetero
    The model paramter is assumed to be fixed at the best one (no estimator). 

    Args:
        CGofTest (UnnormalizedDensity): an instance of UnnormalizedDensity
        alpha (float): significance level
    """

    def __init__(self, p, alpha):
        super(ZhengCDFTest, self).__init__(p, alpha)
        if not(
            type(p) is not cd.CDGaussianOLS
            or type(p) is not cd.CDGaussianHetero
        ):
                raise ValueError(('The given density type {} is not '
                                  'supported'.format(type(p))))
    
    def _handle_cdf_ols(self, X):
        n = X.shape[0]
        p = self.p
        assert type(p) is cd.CDGaussianOLS
        mean = X @ p.slope + p.c
        std = self.p.variance**0.5
        cdfs = [dists.Normal(mean[i], std).cdf
                for i in range(n)] 
        return cdfs

    def _handle_cdf_Gausshetero(self, X):
        n = X.shape[0]
        p = self.p
        assert type(p) is cd.CDGaussianHetero
        mean = p.f(X)
        std = torch.sqrt(p.f_variance(X))
        cdfs = [dists.Normal(mean[i], std[i]).cdf
                 for i in range(n)]
        return cdfs

    def _cdfs_cond_on(self, X):
        return {
            cd.CDGaussianOLS: self._handle_cdf_ols,
            cd.CDGaussianHetero: self._handle_cdf_Gausshetero,
        }.get(type(self.p), lambda x: None)(X)
    
    def eval_cdf(self, X, Y):
        """Returns a matrix whose (i, j) element is CDF F(Y_i|X_j).
        Assuming dy = 1.
        """
        assert self.p.dy() == 1

        n = X.shape[0]
        cdfs = self._cdfs_cond_on(X)

        F = torch.zeros(n, n)
        for j in range(n):
            cdf = cdfs[j]
            F[:, j] = cdf(Y).squeeze()
        return F

    def compute_stat(self, X, Y, h=None):
        """Compute the test static

        Args:
            X (torch.Tensor): X sample 
            Y (torch.Tensor)): Y sample 
            h (torch.Tensor, optional): kernel bandwidth. Defaults to None.
        """
        n, dx = X.shape
        if h is None:
            std = torch.std(X, dim=0)
            h = std * n**(-1./(6*dx))

        Y_pair = CramerVonMisesTest.pairwise_comparison(Y, Y)
        F = self.eval_cdf(X, Y)
        #k = ZhengKLTest.K1
        from math import pi
        k = lambda X: (2.*pi)**(-dx/2.) * torch.exp(-torch.sum(X**2, dim=-1)/2.)
        KX = k((X.unsqueeze(1)-X)/h)
        Diff = Y_pair - F.T

        stat = (KX * (Diff@Diff.T)).fill_diagonal_(0.)
        stat = torch.sum(stat)
        stat = stat * h.prod()**0.5 / (n*(n-1))

        var = ((KX*(Diff@Diff.T)/n)**2).fill_diagonal_(0.)
        var = torch.sum(var)
        var = var * 2. / (n*(n-1)) 
        stat = stat / var**0.5
        return stat
    
    def perform_test(self, X, Y):
        with util.ContextTimer() as t:
            alpha = self.alpha
            stat = self.compute_stat(X, Y)
            pvalue = (1 - dists.Normal(0, 1).cdf(stat)).item()
        results = {'alpha': self.alpha, 'pvalue': pvalue,
                    'test_stat': stat.item(),
                    'h0_rejected': pvalue < alpha, 'time_secs': t.secs,
                    }
        return results
