"""
A module containing kernel functions.
"""

from abc import ABCMeta, abstractmethod
import torch
import kgof
import kgof.kernel as gofker


class Kernel(object):
    """Abstract class for kernels. 
    Inputs to all methods are torch arrays."""

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: nx x d where each row represents one point
        Y: ny x d
        return nx x ny Gram matrix
        """
        raise NotImplementedError()

    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ...

        X: n x d where each row represents one point
        Y: n x d
        return a 1d torch array of length n.
        """
        raise NotImplementedError()


class KCSTKernel(gofker.KSTKernel, Kernel):
    """
    Interface for specifying a Kernel for a Conditional Stein Test (KCST).
    This is infact equivalent to the interface for the unconditional kernel
    Stein test as defined in kgof.kernel.KSTKernel.
    """
    __metaclass__ = ABCMeta
    pass


class HasTunableParams(object):
    """
    Interface specifying that the object has tunable parameters.
    Can be used with a Kernel to have a tunable kernel.
    """
    # constrain_params_function = lambda: None
    _tunable_params = []

    def get_tunable_params(self):
        """
        Return list of parameters to optimize or dicts defining
        parameter groups. Intended to be used with optimizers in `torch.optim`.
        https://pytorch.org/docs/stable/optim.html
        """
        return self._tunable_params

    def set_tunable_params(self, params):
        """
        Set the list of parameters to be returned from
        get_tunable_params()
        """
        self._tunable_params = params

    # def constrain_params(self):
    #     """
    #     Modify the parameters returned from get_tunable_parameters() in-place
    #     so as to satisfy the constraint (if any).
    #     For instance, if a = self.get_tunable_parameters() is a scalar
    #     parameter that must be non-negative, then perhaps what this method
    #     can do in this case is a.clamp_(min=0).
    #     This method will be called after calling optimizer.step().
    #     If there is no constraint, subclasses do not need to implement this.

    #     Return nothing.
    #     """
    #     self.constrain_params_function()


class PTKTestLocations(Kernel):
    """
    A kernel K defined as
    
    K(x, y) = \sum_{i=1}^J k(x, v_i) k(y, v_i)

    for some Kernel k, and a set V = {v_1,..., v_J} containing the test
    locations. This kernel is used in The Finite Set Conditional Discrepancy
    (FSCD). {v_i}_i have to be in the domain that k can accept.
    """

    def __init__(self, k: Kernel, V: torch.tensor):
        """
        k: a base kernel of type Kernel
        V: a torch tensor specifying J locations used to form the kernel
        """
        self.k = k
        self.V = V

        # if isinstance(k, HasTunableParams):
        #     _tunable_params = k.get_tunable_params() + [V]
        # else:
        #     # no tunable parameters in k
        #     _tunable_params = [V]
        # self._tunable_params = _tunable_params

    def eval(self, X, Y):
        k = self.k
        V = self.V
        J = V.shape[0]
        # n x J
        phix = k.eval(X, V)
        phiy = k.eval(Y, V)
        K = phix.matmul(phiy.T)/J
        return K

    def pair_eval(self, X, Y):
        k = self.k
        V = self.V
        J = V.shape[0]
        # n x J
        phix = k.eval(X, V)
        phiy = k.eval(Y, V)
        Kvec = torch.sum(phix*phiy, 1)/J
        return Kvec


class PTKGauss(KCSTKernel):
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    """

    def __init__(self, sigma2):
        """
        sigma2: a number representing the squared bandwidth
        """
        assert sigma2 > 0, 'sigma2 must be > 0. Was %s'%str(sigma2)
        # need to be a tensor to make it tunable with a torch optimizer
        self.sigma2 = torch.tensor([1.0])*sigma2
        # _tunable_params = self.sigma2

    def eval(self, X, Y):
        """
        Evaluate the Gaussian kernel on the two 2d Torch Tensors

        Parameters
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        sigma2 = self.sigma2
        sumx2 = torch.sum(X**2, dim=1).view(-1, 1)
        sumy2 = torch.sum(Y**2, dim=1).view(1, -1)
        D2 = sumx2 - 2.0*torch.matmul(X, Y.transpose(1, 0)) + sumy2
        K = torch.exp(-D2/(2.0*sigma2))
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d Pytorch tensors

        Return
        -------
        a Torch tensor with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = torch.sum( (X-Y)**2, 1)
        sigma2 = torch.sqrt(self.sigma2**2)
        Kvec = torch.exp(-D2/(2.0*sigma2))
        return Kvec

    def __str__(self):
        return "PTKGauss(%.3f)" % self.sigma2

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a Torch tensor of size nx x ny.
        """
        sigma2 = self.sigma2
        K = self.eval(X, Y)
        Diff = X[:, [dim]] - Y[:, [dim]].T
        #Diff = np.reshape(X[:, dim], (-1, 1)) - np.reshape(Y[:, dim], (1, -1))
        G = -K*Diff/sigma2
        return G

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of Y in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a Torch tensor of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny Torch tensor of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        sigma2 = self.sigma2
        D2 = torch.sum(X**2, 1).view(n1, 1) - 2*torch.matmul(X, Y.T) + torch.sum(Y**2, 1).view(1, n2)
        K = torch.exp(-D2/(2.0*sigma2))
        G = K/sigma2 *(d - D2/sigma2)
        return G

    def numpy(self):
        """
        Return an instance of numpy implementation of itself. 
        """
        sigma2 = self.sigma2.item()

        return gofker.KGauss(sigma2)

# end PTKGauss


class KTwoProduct(gofker.Kernel):
    """
    The product of two kernels defined over the tuple of Eucleadean spaces of 
    respective dimensions d1 and d2.

    Args: 
        k1: kgof.Kernel object. 
        k2: kgof.Kernel object. 
        d1: dimensionality of the input of the first kernel 
        d2: dimensionality of the input of the second kernel
    """

    def __init__(self, k1, k2, d1, d2):
        self.k1 = k1
        self.k2 = k2
        self.d1 = d1
        self.d2 = d2

    def eval(self, X, Y):
        """
        Evaluate the kernel on data X and Y
        X: numpy array of size n x (d1+d2) 
        Y: numpy array of size n x (d1+d2)
        Return:  
        """
        d1 = self.d1
        K1 = self.k1.eval(X[:, :d1], Y[:, :d1])
        K2 = self.k2.eval(X[:, d1:], Y[:, d1:])
        return K1 * K2

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        X: numpy array of size n x (d1+d2)
        Y: numpy array of size n x (d1+d2)
        Return: 1d numpy array of length n
        """
        d1 = self.d1
        K1 = self.k1.pair_eval(X[:, :d1], Y[:, :d1])
        K2 = self.k2.pair_eval(X[:, d1:], Y[:, d1:])
        return K1 * K2
