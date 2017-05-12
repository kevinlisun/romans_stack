#!/usr/bin/env python

from svmutil import *
import numpy as np

X = np.random.rand(200,4096)
Y = np.concatenate( (np.ones([50]), 2*np.ones([50]), 3*np.ones([50]), 4*np.ones([50])) )

print X.shape
print Y.shape

prob  = svm_problem( Y.tolist(), X.tolist() )

param = svm_parameter('-t 0 -c 4 -b 1')

model = svm_train(prob, param)

