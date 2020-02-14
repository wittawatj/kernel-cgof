"""Module containing convenient functions for plotting"""

import kcgof
import kcgof.cdensity as cden
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_2d_cond_model(p, px, X, Y, domX, domY, figsize=(10, 6), height_ratios=[3,1], levels=20,
    cmap='pink_r', **contourop ):
    """
    Plot the conditional density model p(y|x) along with the data on a 2d plot.
    Both x, and y must be scalar-valued. 

    p: cdensity.UnnormalizedCondDensity object representing the model p(y|x)
    px: a torch callable that evaluates the density of x
    X, Y: n x 1 torch tensors for the data of x and y
    domX: n x 1 torch tensor specifying points to plot in the domain of x
    domY: n x 1 torch tensor specifying points to plot in the domain of y
    heigh_ratios: height ratios of the subplots. Length of the list specifies
        the number of subplots as well. 2 by default. can be more if more axes
        are needed to plot something else further after returning from this
        function.

    Return (matplotlib figure, axes)
    """
    mdomX, mdomY = torch.meshgrid(domX.view(-1), domY.view(-1))
    flatlogden = p.log_normalized_den(mdomX.reshape(-1, 1), mdomY.reshape(-1, 1))
    flatden = torch.exp(flatlogden)
    mden = flatden.view(mdomX.shape)

    np_mdomX = mdomX.detach().numpy()
    np_mdomY = mdomY.detach().numpy()
    np_mden = mden.detach().numpy()

    # https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
    n_axes = len(height_ratios)
    f, axes = plt.subplots(n_axes, 1, gridspec_kw={'height_ratios': height_ratios}, sharex=True)
    ax1 = axes[0]
    f.set_size_inches(figsize)
    ax1.contourf(np_mdomX, np_mdomY, np_mden, levels=levels, cmap=cmap,
        **contourop)
    # ax1.set_xticklabels([])
    npX = X.detach().numpy()
    npY = Y.detach().numpy()
    ax1.plot(npX, npY, 'bo')
    ax1.set_ylabel('$p(y|x)$')
    # ax1.grid(True)
    # plt.colorbar()

    # plot the px density. Share the horizontal axis
    # ax2 = plt.subplot(212, sharex=ax1)
    # evaluate the px density
    pxden = px(domX)
    np_pxden = pxden.detach().numpy()
    np_domX = domX.detach().numpy()
    ax2 = axes[1]
    ax2.plot(np_domX, np_pxden, 'r-', label='$r_x(x)$')
    ax2.set_xlim(np.min(np_domX), np.max(np_domX))
    # ax2.set_ylabel('$p(x)$')
    ax2.set_xlabel('$x$')
    # plt.legend()
    return f, axes



def plot_2d_cond_data(X, Y):
    """
    X, Y: n x 1 torch tensors for the data of x and y
    """
    npX = X.detach().numpy()
    npY = Y.detach().numpy()
    plt.plot(npX, npY, 'bo')

