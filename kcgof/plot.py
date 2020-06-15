"""Module containing convenient functions for plotting"""

import kcgof
import kcgof.cdensity as cden
import kcgof.glo as glo

import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_func_tuples():
    """
    Return a list of tuples where each tuple is of the form
        (func_name used in the experiments, label name, plot line style)
    """
    func_tuples = [
        ('met_cramer_vm', 'Cramer', 'k-3'),
        ('met_gmmd_med', 'MMD-med', 'g-+'),
        ('met_gmmd_split_med', 'MMD', 'g-<'),

        ('met_zhengkl_mc', 'Zheng MC', 'm-'),
        ('met_zhengkl_gh', 'Zheng GH', 'm--'),
        ('met_zheng_cdf', 'Zheng CDF', 'm:'),

        ('met_gkssd_med', 'KSSD', 'r-'),
        ('met_gkssd_opt_tr30', 'KSSD-opt 30tr', 'r-2'),
        ('met_gkssd_opt_tr50', 'KSSD-opt 50tr', 'r-^'),
        
        ('met_gfscd_J1_rand', 'FSCD-rand J1', 'c--D'),
        ('met_gfscd_J5_rand', 'FSCD-rand J5', 'k-.<'),
        ('met_gfscd_J1_opt_tr30', 'FSCD-opt J1', 'm-x'),
        ('met_gfscd_J1_opt_tr50', 'FSCD-opt J1 50tr', 'b-o'),
        ('met_gfscd_J5_opt_tr30', 'FSCD-opt J5', 'b-P'),
        ('met_gfscd_J5_opt_tr50', 'FSCD-opt J5 50tr', 'b-s'),

        ('met_zhengkl', 'Zheng', 'm-x'),
        # ('met_gumeJ5_3sopt_tr20', 'Rel-UME J5', 'r-^'),
        # ('met_gfssdJ1_3sopt_tr20', 'Rel-FSSD J1', 'C4--'),
        # ('met_gfssdJ5_3sopt_tr20', 'Rel-FSSD J5', 'b-x'),

        # ('met_gmmd_med', 'Rel-MMD', 'k-.'),
        # ('met_gmmd_med_bounliphone', 'Rel-MMD medboun', 'k-'),

        # ('met_gfssdJ1_3sopt_tr50', 'FSSD-opt3 J1', 'b-^'),
        # ('met_gfssdJ5_3sopt_tr50', 'FSSD-opt3 J5', 'b-.h'),

        # ('met_gumeJ1_2V_rand', 'UME-rand J1', 'r--^'),
        # ('met_gumeJ1_1V_rand', 'UME-rand J1 1V', 'y-'),
        # ('met_gumeJ2_2V_rand', 'UME-rand J2', 'g--^'),
        # ('met_gumeJ3_2V_rand', 'UME-rand J3', 'b--^'),
        # ('met_gumeJ5_2V_rand', 'UME-rand J5', 'k--^'),

        # ('met_gumeJ1_2sopt_tr20', 'Rel-UME-opt2 J1', 'C2-.'),
        # ('met_gumeJ5_2sopt_tr20', 'Rel-UME-opt2 J5', 'g-'),
        # ('met_gumeJ1_2sopt_tr50', 'Rel-UME-opt2 J1', 'r-.h'),

        # ('met_gumeJ1_3sopt_tr50', 'UME-opt3 J1', 'r-'),
        # ('met_gumeJ5_3sopt_tr50', 'UME-opt3 J5', 'k-'),
        ]
    return func_tuples

def get_func2label_map():
    # map: job_func_name |-> plot label
    func_tuples = get_func_tuples()
    #M = {k:v for (k,v) in zip(func_names, labels)}
    M = {k:v for (k,v,_) in func_tuples}
    return M

def func_plot_fmt_map():
    """
    Return a map from job function names to matplotlib plot styles 
    """
    # line_styles = ['o-', 'x-',  '*-', '-_', 'D-', 'h-', '+-', 's-', 'v-', 
    #               ',-', '1-']
    func_tuples = get_func_tuples()
    M = {k:v for (k, _, v) in func_tuples}
    return M


def plot_prob_reject(ex, fname, func_xvalues, xlabel, func_title=None, 
        return_plot_values=False):
    """
    plot the empirical probability that the statistic is above the threshold.
    This can be interpreted as type-1 error (when H0 is true) or test power 
    (when H1 is true). The plot is against the specified x-axis.

    - ex: experiment number 
    - fname: file name of the aggregated result
    - func_xvalues: function taking aggregated results dictionary and return the values 
        to be used for the x-axis values.            
    - xlabel: label of the x-axis. 
    - func_title: a function: results dictionary -> title of the plot
    - return_plot_values: if true, also return a PlotValues as the second
      output value.

    Return loaded results
    """
    #from IPython.core.debugger import Tracer 
    #Tracer()()

    results = glo.ex_load_result(ex, fname)

    def rej_accessor(jr):
        rej = jr['test_result']['h0_rejected']
        # When used with vectorize(), making the value float will make the resulting 
        # numpy array to be of float. nan values can be stored.
        return float(rej)

    #value_accessor = lambda job_results: job_results['test_result']['h0_rejected']
    vf_pval = np.vectorize(rej_accessor)
    # results['job_results'] is a dictionary: 
    # {'test_result': (dict from running perform_test(te) '...':..., }
    rejs = vf_pval(results['job_results'])
    repeats, _, n_methods = results['job_results'].shape

    # yvalues (corresponding to xvalues) x #methods
    mean_rejs = np.mean(rejs, axis=0)
    #print mean_rejs
    #std_pvals = np.std(rejs, axis=0)
    #std_pvals = np.sqrt(mean_rejs*(1.0-mean_rejs))

    xvalues = func_xvalues(results)

    #ns = np.array(results[xkey])
    #te_proportion = 1.0 - results['tr_proportion']
    #test_sizes = ns*te_proportion
    line_styles = func_plot_fmt_map()
    method_labels = get_func2label_map()
    
    func_names = [f.__name__ for f in results['method_funcs'] ]
    plotted_methods = []
    for i in range(n_methods):    
        #te_proportion = 1.0 - results['tr_proportion']
        fmt = line_styles[func_names[i]]
        #plt.errorbar(ns*te_proportion, mean_rejs[:, i], std_pvals[:, i])
        method_label = method_labels[func_names[i]]
        plotted_methods.append(method_label)
        plt.plot(xvalues, mean_rejs[:, i], fmt, label=method_label)
    '''
    else:
        # h0 is true 
        z = stats.norm.isf( (1-confidence)/2.0)
        for i in range(n_methods):
            phat = mean_rejs[:, i]
            conf_iv = z*(phat*(1-phat)/repeats)**0.5
            #plt.errorbar(test_sizes, phat, conf_iv, fmt=line_styles[i], label=method_labels[i])
            plt.plot(test_sizes, mean_rejs[:, i], line_styles[i], label=method_labels[i])
    '''
            
    ylabel = 'Rejection rate'
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(np.hstack((xvalues) ))
    
    alpha = results['alpha']
    plt.legend(loc='best')
    title = '%s. %d trials. $\\alpha$ = %.2g.'%( results['prob_label'],
            repeats, alpha) if func_title is None else func_title(results)
    plt.title(title)
    plt.grid()
    if return_plot_values:
        return results, PlotValues(xvalues=xvalues, methods=plotted_methods,
                plot_matrix=mean_rejs.T)
    else:
        return results

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
    # for the purpose of plotting, if the density is Nan, make it 0
    flatden[torch.isnan(flatden)] = 0.0
    # print(flatden)
    # print(torch.all(torch.abs(flatden) <= 1e-7))
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
    ax1.plot(npX, npY, 'bo', markersize=4)
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
