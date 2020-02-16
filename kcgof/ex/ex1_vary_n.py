"""Simulation to get the test power vs increasing sample size"""

__author__ = 'wittawat'

import kcgof
import kcgof.log as log
import kcgof.glo as glo
import kcgof.cdata as cdat
import kcgof.cgoftest as cgof
import kcgof.cdensity as cden
import kcgof.kernel as ker
import kcgof.util as util

# need independent_jobs package 
# https://github.com/wittawatj/independent-jobs
import independent_jobs as inj
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.tools.Log import logger
import numpy as np
import math
import os
import sys 
import scipy
import scipy.stats as stats
import torch

"""
All the method functions (starting with met_) return a dictionary with the
following keys:
    - test: test object. (may or may not return to save memory)
    - test_result: the result from calling perform_test(te).
    - time_secs: run time in seconds 

    * A method function may return an empty dictionary {} if the inputs are not
    applicable. 

All the method functions take the following mandatory inputs:
    - p: a kcgof.cdensity.UnnormalizedCondDensity (candidate conditional model)
    - rx: an object that can be sample(..)'ed. Following the interface of a
        distribution in torch.distributions.
    - cond_source: a kcgof.cdata.CondSource for generating the data (i.e., draws
          from r)
    - n: total sample size. Each method function should draw exactly the number
          of points from the cond_source.
    - r: repetition (trial) index. Drawing samples should make use of r to
          set the random seed.
    -------
    - A method function may have more arguments which have default values.
"""
def sample_xy(rx, cond_source, n, rep):
    """
    rep: repetition/trial index
    All methods should call this to get the exact same data X,Y in each trial.
    """
    cs = cond_source
    r = rep
    with util.TorchSeedContext(seed=r):
        X = rx(n)
    Y = cs(X, seed=r+100)
    return (X, Y)


#-------------------------------------------------------
def met_gkssd_med(p, rx, cond_source, n, r):
    """
    KSSD test with Gaussian kernels (for both kernels). Prefix g = Gaussian kernel.
    med = Use median heuristic to choose the bandwidths for both kernels.
    Compute the median heuristic on the data X and Y separate to get the two
    bandwidths.
    """
    X, Y = sample_xy(rx, cond_source, n, r)

    # start timing
    with util.ContextTimer() as t:
        # median heuristic
        sigx = util.pt_meddistance(X, subsample=1000)
        sigy = util.pt_meddistance(Y, subsample=1000)

        # kernels
        # k = kernel on X
        k = ker.PTKGauss(sigma2=sigx**2)
        # l = kernel on Y
        l = ker.PTKGauss(sigma2=sigy**2)

        # Construct a KSSD test object
        kssdtest = cgof.KSSDTest(p, k, l, alpha=alpha, n_bootstrap=400, seed=r+88)
        result = kssdtest.perform_test(X, Y)

    return { 'test': kssdtest,
        'test_result': result, 'time_secs': t.secs}

def met_gfscd_J5_rand(p, rx, cond_source, n, r):
    return met_gfscd_J1_rand(p, rx, cond_source, n, r, J=5)

def met_gfscd_J1_rand(p, rx, cond_source, n, r, J=1):
    """
    FSCD test with Gaussian kernels on both X and Y.
    * Use J=1 random test location by default.
    * The test locations are drawn from a Gaussian fitted to the data drawn
        from rx.
    * Bandwithds of the Gaussian kernels are determined by the median
        heuristic.
    """
    X, Y = sample_xy(rx, cond_source, n, r)
    # start timing
    with util.ContextTimer() as t:
        # fit a Gaussian and draw J locations
        npV = util.fit_gaussian_sample(X.detach().numpy(), J, seed=r+55)
        V = torch.tensor(npV, dtype=torch.float)

        # median heuristic
        sigx = util.pt_meddistance(X, subsample=1000)
        sigy = util.pt_meddistance(Y, subsample=1000)

        # kernels
        # k = kernel on X
        k = ker.PTKGauss(sigma2=sigx**2)
        # l = kernel on Y
        l = ker.PTKGauss(sigma2=sigy**2)

        # Construct a FSCD test object
        fscdtest = cgof.FSCDTest(p, k, l, V, alpha=alpha, n_bootstrap=400, seed=r+8)
        result = fscdtest.perform_test(X, Y)

    return { 'test': fscdtest,
        'test_result': result, 'time_secs': t.secs}

def met_gfscd_J5_opt_tr50(p, rx, cond_source, n, r):
    return met_gfscd_J1_opt_tr50(p, rx, cond_source, n, r, J=5, tr_proportion=0.5)

def met_gfscd_J1_opt_tr50(p, rx, cond_source, n, r, J=1, tr_proportion=0.5):
    """
    FSCD test with Gaussian kernels on both X and Y.
    Optimize both Gaussian bandwidhts and the test locations by maximizing
    the test power.
    The proportion of the training data used for the optimization is
    controlled by tr_proportion.
    """
    X, Y = sample_xy(rx, cond_source, n, r)
    # start timing
    with util.ContextTimer() as t:
        # fit a Gaussian and draw J locations as an initial point for V
        npV = util.fit_gaussian_sample(X.detach().numpy(), J, seed=r+55)
        V = torch.tensor(npV, dtype=torch.float)

        # median heuristic
        sigx = util.pt_meddistance(X, subsample=1000)
        sigy = util.pt_meddistance(Y, subsample=1000)

        # kernels
        # k = kernel on X
        k = ker.PTKGauss(sigma2=sigx**2)
        # l = kernel on Y
        l = ker.PTKGauss(sigma2=sigy**2)

        # split the data 
        cd = cdat.CondData(X, Y)
        tr, te = cd.split_tr_te(tr_proportion=tr_proportion)

        # training data
        Xtr, Ytr = tr.xy()
        abs_min, abs_max = torch.min(Xtr).item(), torch.max(Xtr).item()
        abs_std = torch.std(Xtr).item()

        # parameter tuning
        fscd_pc = cgof.FSCDPowerCriterion(p, k, l, Xtr, Ytr)
        max_iter = 200
        # learning rate
        lr = 1e-2
        # regularization parameter when forming the power criterion
        reg = 1e-3

        # constraint satisfaction function
        def con_f(params, V):
            ksigma2 = params[0]
            lsigma2 = params[1]
            ksigma2.data.clamp_(min=1e-2, max=10*sigx**2)
            lsigma2.data.clamp_(min=1e-2, max=10*sigy**2)
            V.data.clamp_(min=abs_min - 2.0*abs_std, max=abs_max + 2.0*abs_std)

        # do the optimization. Parameters are optimized in-place
        fscd_pc.optimize_params([k.sigma2, l.sigma2], V, constraint_f=con_f,
            lr=lr, reg=reg, max_iter=max_iter)

        # Now that k, l, and V are optimized. Construct a FSCD test object
        fscdtest = cgof.FSCDTest(p, k, l, V, alpha=alpha, n_bootstrap=400, seed=r+8)
        result = fscdtest.perform_test(X, Y)

    return { 'test': fscdtest,
        'test_result': result, 'time_secs': t.secs}

def met_zhengkl(p, rx, cond_source, n, r):
    """
    "Zheng 2000, A CONSISTENT TEST OF CONDITIONAL PARAMETRIC DISTRIBUTIONS", 
    which uses the first order approximation of KL divergence as the decision
    criterion. 
    Use cgoftest.ZhengKLTest.
    """
    X, Y = sample_xy(rx, cond_source, n, r)
    # start timing
    with util.ContextTimer() as t:

        # the test
        zheng_test = cgof.ZhengKLTest(p, alpha)
        result = zheng_test.perform_test(X, Y)

    return { 'test': zheng_test,
        'test_result': result, 'time_secs': t.secs}

# def met_gmmd_med(P, Q, data_source, n, r):
#     """
#     Use met_gmmd_med_bounliphone(). It uses the median heuristic following
#     Bounliphone et al., 2016.

#     Bounliphone et al., 2016's MMD-based 3-sample test.
#     * Gaussian kernel. 
#     * Gaussian width = mean of (median heuristic on (X, Z), median heuristic on
#         (Y, Z))
#     * Use full sample for testing (no
#     holding out for optimization)
#     """
#     if not P.has_datasource() or not Q.has_datasource():
#         # Not applicable. Return {}.
#         return {}

#     ds_p = P.get_datasource()
#     ds_q = Q.get_datasource()
#     # sample some data 
#     datp, datq, datr = sample_pqr(ds_p, ds_q, data_source, n, r, only_from_r=False)

#     # Start the timer here
#     with util.ContextTimer() as t:
#         X, Y, Z = datp.data(), datq.data(), datr.data()

#         # hyperparameters of the test
#         medxz = util.meddistance(np.vstack((X, Z)), subsample=1000)
#         medyz = util.meddistance(np.vstack((Y, Z)), subsample=1000)
#         medxyz = np.mean([medxz, medyz])
#         k = kernel.KGauss(sigma2=medxyz**2)

#         scmmd = mct.SC_MMD(datp, datq, k, alpha=alpha)
#         scmmd_result = scmmd.perform_test(datr)

#     return {
#             # This key "test" can be removed.             
#             #'test': scmmd, 
#             'test_result': scmmd_result, 'time_secs': t.secs}

# Define our custom Job, which inherits from base class IndependentJob
class Ex1Job(IndependentJob):
   
    def __init__(self, aggregator, p, rx, cond_source, prob_label, rep, met_func, n):
        #walltime = 60*59*24 
        walltime = 60*59
        memory = int(n*1e-2) + 50

        IndependentJob.__init__(self, aggregator, walltime=walltime,
                               memory=memory)
                               
        self.p = p
        self.rx = rx
        self.cond_source = cond_source
        self.prob_label = prob_label
        self.rep = rep
        self.met_func = met_func
        self.n = n

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):

        p = self.p
        rx = self.rx
        cs = self.cond_source 
        r = self.rep
        n = self.n
        met_func = self.met_func
        prob_label = self.prob_label

        logger.info("computing. %s. prob=%s, r=%d,\
                n=%d"%(met_func.__name__, prob_label, r, n))
        with util.ContextTimer() as t:
            job_result = met_func(p, rx, cs, n, r)

            # create ScalarResult instance
            result = SingleResult(job_result)
            # submit the result to my own aggregator
            self.aggregator.submit_result(result)
            func_name = met_func.__name__

        logger.info("done. ex2: %s, prob=%s, r=%d, n=%d. Took: %.3g s "%(func_name,
            prob_label, r, n, t.secs))

        # save result
        fname = '%s-%s-n%d_r%d_a%.3f.p' \
                %(prob_label, func_name, n, r, alpha )
        glo.ex_save_result(ex, job_result, prob_label, fname)

# This import is needed so that pickle knows about the class Ex1Job.
# pickle is used when collecting the results from the submitted jobs.
from kcgof.ex.ex1_vary_n import Ex1Job
from kcgof.ex.ex1_vary_n import met_gkssd_med
from kcgof.ex.ex1_vary_n import met_zhengkl
from kcgof.ex.ex1_vary_n import met_gfscd_J1_rand
from kcgof.ex.ex1_vary_n import met_gfscd_J5_rand
from kcgof.ex.ex1_vary_n import met_gfscd_J1_opt_tr50
from kcgof.ex.ex1_vary_n import met_gfscd_J5_opt_tr50

#--- experimental setting -----
ex = 1

# significance level of the test
alpha = 0.05

# Proportion of training sample relative to the full sample size n. 
# Only used by tests that do data splitting for parameter optimization.
# tr_proportion = 0.5

# repetitions for each sample size 
reps = 100

# tests to try
method_funcs = [ 
    met_gkssd_med,
    # met_zhengkl,
    met_gfscd_J1_rand,
    met_gfscd_J1_opt_tr50,
   ]

# If is_rerun==False, do not rerun the experiment if a result file for the current
# setting already exists.
is_rerun = False
#---------------------------

def get_ns_model_source(prob_label):
    """
    Given the problem key prob_label, return (ns, p, cs), a tuple of
    - ns: a list of sample sizes n's
    - p: a kcgof.cdensity.UnnormalizedCondDensity representing the model p
    - rx: a callable object that takes n (sample size) and return a torch
        tensor of size n x d where d is the appropriate dimension. Represent the
        marginal distribuiton of x
    - cs: a kcgof.cdata.CondSource. The CondSource generates sample from the
        distribution r.

    * (p, cs) together specifies a conditional goodness-of-fit testing problem.
    """
    slope_h0_d5 = torch.arange(5) + 1.0
    # slope_h0_d20 = torch.arange(20) + 1.0
    prob2tuples = { 
        # A case where H0 is true. Gaussian least squares model.
        'gaussls_h0_d5': (
            [100, 300, 600, 900],
            # p 
            cden.CDGaussianOLS(slope=slope_h0_d5, c=0, variance=1.0),
            # rx
            cden.RXIsotropicGaussian(dx=5),
            # CondSource for r]
            cdat.CSGaussianOLS(slope=slope_h0_d5, c=0, variance=1.0),
            ),

        # 'gmm_blobs_d2_dfac4': 
        #     # list of sample sizes
        #     ([300, 600, 900, 1200, 1500], ) + make_gmm_blobs_d2(distance_factor=4.0),

        } # end of prob2tuples
    if prob_label not in prob2tuples:
        raise ValueError('Unknown problem label. Need to be one of %s'%str(list(prob2tuples.keys()) ))
    return prob2tuples[prob_label]


def run_problem(prob_label, use_cluster=False):
    """Run the experiment"""
    # ///////  submit jobs //////////
    # create folder name string
    #result_folder = glo.result_folder()
    from kcgof.config import get_default_config
    config = get_default_config()
    tmp_dir = config['ex_scratch_path']
    foldername = os.path.join(tmp_dir, 'kcgof_slurm', 'e%d'%ex)
    logger.info("Setting engine folder to %s" % foldername)

    # create parameter instance that is needed for any batch computation engine
    logger.info("Creating batch parameter instance")
    batch_parameters = BatchClusterParameters(
        foldername=foldername, job_name_base="e%d_"%ex, parameter_prefix="")

    if use_cluster:
        # use a Slurm cluster
        partitions = config['ex_slurm_partitions']
        if partitions is None:
            engine = SlurmComputationEngine(batch_parameters)
        else:
            engine = SlurmComputationEngine(batch_parameters, partition=partitions)
    else:
        # Use the following line if Slurm queue is not used.
        engine = SerialComputationEngine()
    n_methods = len(method_funcs)

    # problem setting
    ns, p, rx, cs = get_ns_model_source(prob_label)

    # repetitions x len(ns) x #methods
    aggregators = np.empty((reps, len(ns), n_methods ), dtype=object)

    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                # name used to save the result
                func_name = f.__name__
                fname = '%s-%s-n%d_r%d_a%.3f.p' \
                        %(prob_label, func_name, n, r, alpha,)
                if not is_rerun and glo.ex_file_exists(ex, prob_label, fname):
                    logger.info('%s exists. Load and return.'%fname)
                    job_result = glo.ex_load_result(ex, prob_label, fname)

                    sra = SingleResultAggregator()
                    sra.submit_result(SingleResult(job_result))
                    aggregators[r, ni, mi] = sra
                else:
                    # result not exists or rerun
                    job = Ex1Job(SingleResultAggregator(), p, rx, cs,
                        prob_label, r, f, n)

                    agg = engine.submit_job(job)
                    aggregators[r, ni, mi] = agg

    # let the engine finish its business
    logger.info("Wait for all call in engine")
    engine.wait_for_all()

    # ////// collect the results ///////////
    logger.info("Collecting results")
    job_results = np.empty((reps, len(ns), n_methods), dtype=object)
    for r in range(reps):
        for ni, n in enumerate(ns):
            for mi, f in enumerate(method_funcs):
                logger.info("Collecting result (%s, r=%d, n=%d)" %
                        (f.__name__, r, n))
                # let the aggregator finalize things
                aggregators[r, ni, mi].finalize()

                # aggregators[i].get_final_result() returns a SingleResult instance,
                # which we need to extract the actual result
                job_result = aggregators[r, ni, mi].get_final_result().result
                job_results[r, ni, mi] = job_result

    #func_names = [f.__name__ for f in method_funcs]
    #func2labels = exglobal.get_func2label_map()
    #method_labels = [func2labels[f] for f in func_names if f in func2labels]

    # save results 
    results = {'job_results': job_results, 
            'p': p, 
            'cond_source': cs, 
            'alpha': alpha, 'repeats': reps, 'ns': ns,
            'method_funcs': method_funcs, 'prob_label': prob_label,
            }
    
    # class name 
    fname = 'ex%d-%s-me%d_rs%d_nmi%d_nma%d_a%.3f.p' \
        %(ex, prob_label, n_methods, reps, min(ns), max(ns), alpha,)

    glo.ex_save_result(ex, results, fname)
    logger.info('Saved aggregated results to %s'%fname)

#---------------------------
def main():
    if len(sys.argv) != 2:
        print('Usage: %s problem_label'%sys.argv[0])
        sys.exit(1)
    prob_label = sys.argv[1]
    run_problem(prob_label)

if __name__ == '__main__':
    main()

