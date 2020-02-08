"""
Module for testing cdensity module.
"""

__author__ = 'wittawat'

import numpy as np
import matplotlib.pyplot as plt
import kcgof
import kcgof.cdensity as cden
import kcgof.glo as glo
import scipy.stats as stats

import unittest

# class TestIsotropicNormal(unittest.TestCase):
#     def setUp(self):
#         pass


#     def test_log_den(self):
#         n = 7
#         with util.NumpySeedContext(seed=16):
#             for d in [3, 1]:
#                 variance = 1.1
#                 mean = np.random.randn(d)
#                 X = np.random.rand(n, d) + 1

#                 isonorm = density.IsotropicNormal(mean, variance)
#                 log_dens = isonorm.log_den(X)
#                 my_log_dens = -np.sum((X-mean)**2, 1)/(2.0*variance)

#                 # check correctness 
#                 np.testing.assert_almost_equal(log_dens, my_log_dens)

#     def test_grad_log(self):
#         n = 8
#         with util.NumpySeedContext(seed=17):
#             for d in [4, 1]:
#                 variance = 1.2
#                 mean = np.random.randn(d) + 1
#                 X = np.random.rand(n, d) - 2 

#                 isonorm = density.IsotropicNormal(mean, variance)
#                 grad_log = isonorm.grad_log(X)
#                 my_grad_log = -(X-mean)/variance

#                 # check correctness 
#                 np.testing.assert_almost_equal(grad_log, my_grad_log)

#     def tearDown(self):
#         pass



if __name__ == '__main__':
   unittest.main()

