#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-11-03 18:35:52
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2016-11-08 16:53:30


import caffe
import score
import numpy as np


weights = '/home/kevin/snapshot/caffe-net-full.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

test_list = np.loadtxt('/home/kevin/dataset/processed_data/test_object.txt', dtype=str)
# test_list = test_list[::10,:]

for i in range(15):
	solver.step(9708)
	score.model_classification_test(solver, '/home/kevin/tmp/'.format(i), test_list, layer='fc8', gt='label')
