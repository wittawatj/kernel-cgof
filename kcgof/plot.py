"""Module containing convenient functions for plotting"""

import kcgof
import kcgof.cdensity as cden
import matplotlib 
import matplotlib.pyplot as plt
import torch

def plot_2d_cond_model(p, px, X, Y, domX, domY, figsize=(10, 6), levels=20, cmap='pink_r', **contourop ):
    """
    Plot the conditional density model p(y|x) along with the data on a 2d plot.
    Both x, and y must be scalar-valued. 

    p: cdensity.UnnormalizedCondDensity object representing the model p(y|x)
    px: a torch callable that evaluates the density of x
    X, Y: n x 1 torch tensors for the data of x and y
    domX: n x 1 torch tensor specifying points to plot in the domain of x
    domY: n x 1 torch tensor specifying points to plot in the domain of y
    """
    mdomX, mdomY = torch.meshgrid(domX.view(-1), domY.view(-1))
    flatlogden = p.log_normalized_den(mdomX.reshape(-1, 1), mdomY.reshape(-1, 1))
    flatden = torch.exp(flatlogden)
    mden = flatden.view(mdomX.shape)

    np_mdomX = mdomX.detach().numpy()
    np_mdomY = mdomY.detach().numpy()
    np_mden = mden.detach().numpy()

    # https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    f.set_size_inches(figsize)
    ax1.contourf(np_mdomX, np_mdomY, np_mden, levels=levels, cmap=cmap,
        **contourop)
    # ax1.set_xticklabels([])
    npX = X.detach().numpy()
    npY = Y.detach().numpy()
    ax1.plot(npX, npY, 'bo')
    ax1.set_ylabel('$p(y|x)$')
    # plt.colorbar()

    # plot the px density. Share the horizontal axis
    # ax2 = plt.subplot(212, sharex=ax1)
    # evaluate the px density
    pxden = px(domX)
    np_pxden = pxden.detach().numpy()
    ax2.plot(np_mdomX, np_pxden, 'r-')
    ax2.set_ylabel('$p(x)$')
    ax2.set_xlabel('$x$')
    # plt.legend()



def plot_2d_cond_data(X, Y):
    """
    X, Y: n x 1 torch tensors for the data of x and y
    """
    npX = X.detach().numpy()
    npY = Y.detach().numpy()
    plt.plot(npX, npY, 'bo')

