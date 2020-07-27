# kcgof

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/kernel-cgof/blob/master/LICENSE)

This repository contains a Python 3 implementation of two nonparametric
goodness-of-fit testing of conditional density models as proposed in our paper

    Testing Goodness of Fit of Conditional Density Models with Kernels
    Wittawat Jitkrittum, Heishiro Kanagawa, Bernhard Schölkopf
    UAI 2020
    https://arxiv.org/abs/2002.10271

* Presentation slides for UAI 2020 can be found [here](https://docs.google.com/presentation/d/14I4ndHux8C3ImRzAqmNPVLmLQkw8wZQucMMTZrACUXk/edit?usp=sharing).
* Video presentation here (TODO: link not available yet).

## Dependency 

* dill == 0.3.1.1
* matplotlib == 3.1.3
* numpy == 1.18.1
* scipy == 1.4.1
* torch == 1.4.0

Version numbers for everything else except `torch` should not matter much as
long as they are not too old.

## Demo

See the Jupyter notebook [ipynb/demo_proposed_tests.ipynb](https://github.com/wittawatj/kernel-cgof/blob/master/ipynb/demo_proposed_tests.ipynb) for how to start using our tests. This notebook describes how to use the proposed Kernel Conditional Stein Discrepancy (KCSD) test, which is one of the two test we proposed. 

A notebook that introduces the other test, the Finite Set Conditional Discrepancy (FSCD), is coming soon.

## Development

We recommend [Anaconda](https://www.anaconda.com/). To install our package
`kcgof` for development purpose, follow the following steps:

1. Make a new Anaconda environment for this project. We will need Pytorch.
    Switch to this environment.

2. Install `kernel-gof` (dependency). See https://github.com/wittawatj/kernel-gof. 

3. (Optional. Only needed if you use the MMD test in this repository.) Install `freqopttest`. See https://github.com/wittawatj/interpretable-test.

4. Install the above dependencies with `conda install` or `pip install` in your environment.

5. Clone this repository to your local folder.  

6. Run the following command in a terminal to install the `kcgof` package
    from this repository.

        pip install -e /path/to/the/local/folder

7. In a Python shell under the same conda environment, make sure that you can `import kcgof` without any error.

The `-e` flag offers an "edit mode", so that changes to any files in this
repo will be reflected immediately in the imported package.


## Reproduce experimental results


* Batch experiments require https://github.com/wittawatj/independent-jobs .

All experiments which involve test powers can be found in
`kcgof/ex/ex1_vary_n.py`. To run the experiment, use `kcgof/ex/run_ex1.sh`. 

We used [independent-jobs](https://github.com/wittawatj/independent-jobs)
package to parallelize our experiments over a
[Slurm](http://slurm.schedmd.com/) cluster (the package is not needed if you
just need to use our developed tests without reproducing our results). For example, for
`ex1_vary_n.py`, a job is created for each combination of 

    (dataset, test algorithm, n, trial)

If you do not use Slurm, in `kcgof/config.py`, set

    'ex_use_slurm_cluster': False,

which will instruct the computation engine to just use a normal for-loop on a
single machine (will take a lot of time).  Running simulation will
create a lot of result files (one for each tuple above) saved as Pickle. Also, the `independent-jobs`
package requires a scratch folder to save temporary files for communication
among computing nodes. Path to the folder containing the saved results can be specified in 
the same config file by changing the value of `ex_results_path`.
The scratch folder needed by the `independent-jobs` package can be specified in the same file
by changing the value of `ex_scratch_path`.

To plot the results, see the experiment's corresponding Jupyter notebook in the
`ipynb/` folder. For example, for `ex1_vary_n.py` see `ipynb/ex1_results.ipynb` to plot the results.

---------------

If you have questions or comments about anything related to this work, please
do not hesitate to contact [Wittawat Jitkrittum](http://wittawat.com) and
    [Heishiro Kanagawa](https://noukoudashisoup.github.io/)
