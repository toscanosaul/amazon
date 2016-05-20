#!/usr/bin/env python

from recommend.mf.matrix_factorization import MatrixFactorization
import numpy as np

num_users=943
num_item=1682



data=np.loadtxt("ml-100k/u.data")
print data[1,:]