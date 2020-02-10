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
        self.sigma2 = sigma2

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
        D2 = sumx2 - 2*torch.matmul(X, Y.transpose(1, 0)) + sumy2
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

        Return a numpy array of size nx x ny.
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

        Return a numpy array of size nx x ny.
        """
        return -self.gradX_Y(X, Y, dim)

    def gradXY_sum(self, X, Y):
        r"""
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
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

# end PTKGauss
