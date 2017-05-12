#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-12-09 17:04:20
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-02-23 16:41:44

import caffe
from svmutil import *

import GPy

import numpy as np
from PIL import Image
from scipy import io

import os
import time
import sys


def extract_feature( dir ):
	global caffe 
	global net
	global mode

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	feature = np.array( [] )

	for file in os.listdir( dir ):
		if file.endswith( ".jpg" ):
			print( 'extracting feature from ' + dir + '/' + file + '...' )

			rgb = Image.open( dir + '/' + file )
			np_rgb = np.array( rgb, dtype=np.float32 )

			# convert rgb to bgr
			np_rgb = np_rgb[:,:,::-1]
			# substract the mean
			np_rgb -= np.array( (104.00698793,116.66876762,122.67891434) )

			# transpose
			np_rgb = np_rgb.transpose( (2,0,1) )

			# shape for input (data blob is 1 x C x H x W), set data
			net.blobs['data'].reshape(1, *np_rgb.shape)
			net.blobs['data'].data[...] = np_rgb

			start_time = time.time()
			net.forward()
			print(time.time()-start_time)

			# get the feature blob
			feat_fc7 = np.array( net.blobs['fc7'].data, dtype=np.float32 )
			print feat_fc7.shape

			if feature.size == 0:
				feature = feat_fc7
			else:
				feature = np.concatenate( (feature, feat_fc7), 0 )

	return feature

def gp_train(X, Y):

	k = GPy.kern.RBF(X.shape[1],variance=250,lengthscale=250)
	print k.K(X)

	io.savemat('./kernel.mat', {'K':np.array(k.K(X))})

	f = np.random.multivariate_normal(np.zeros(X.shape[0]), k.K(X))
	lik = GPy.likelihoods.Bernoulli()

	m = GPy.core.GP(X=X,
                Y=Y, 
                kernel=k, 
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik)

	for i in range(5):
		m.optimize('bfgs', max_iters=100) #first runs EP and then optimizes the kernel parameters
		print 'iteration:', i,
		print m
		print ""

	print m

	return m



def main():

	global caffe
	global net
	global mode

	#------------------------------------- setting ---------------------------------------------------------
	mode = sys.argv[1]

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	net = caffe.Net('/home/kevin/models/vgg16/deploy.prototxt', '/home/kevin/models/vgg16/model.caffemodel', caffe.TEST)
	
	data_dir = '/home/kevin/catkin_ws/data/labelled/'
	folder = [ 'background', 'bottles', 'cans', 'chains', 'cloth', 'gloves', 'metal_objects', 'pipe_joints', 'plastic_pipes', 'sponges', 'wood_blocks' ]
	label = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

	model_file_name = './svm_model.model'
	#---------------------------------------------------------------------------------------------------------

	instances = np.array( [] )
	labels = np.array( [] )

	for foldi, labeli in zip( folder, label ):

		instance_i = extract_feature( data_dir + foldi ) 

		if instances.size == 0:
			instances = instance_i
			labels = np.ones( [instance_i.shape[0]] ) * labeli
		else:
			instances = np.concatenate( ( instances, instance_i ), 0 )
			labels = np.concatenate( ( labels, np.ones( [instance_i.shape[0]]) * labeli ) )

	# train lib-SVM
	labels = labels.reshape((labels.shape[0], 1))
	print instances.shape
	print labels.shape

	gp_model = gp_train(instances, labels)

	# save GP model to the disk
	X = instances
	Y = labels

	np.save('gp_model_param.npy', gp_model.param_array) 
	np.save('gp_model_X.npy', X)
	np.save('gp_model_Y.npy', Y)  


	# loading a model
	# Model creation, without initialization:
	#gp_model = GPy.models.GPClassification(instances, labels, initialize=False)
	#gp_model.update_model(False) # do not call the underlying expensive algebra on load
	#gp_model.initialize_parameter() # Initialize the parameters (connect the parameters up)
	#gp_model[:] = np.load('gp_model_rgb.npy') # Load the parameters
	#gp_model.update_model(True) # Call the algebra only once
	#print(gp_model)
	#probs = gp_model.predict(instances)[0]
	#print probs

	## train SVM
	#io.savemat('./features.mat', {'instances':np.array(instances), 'labels':np.array(labels)})
	#problem  = svm_problem( labels.tolist(), instances.tolist() )
	#svm_opt = svm_parameter( '-t 0 -c 4' )
	#model = svm_train( problem, svm_opt )
	#svm_save_model( model_file_name, model )



if __name__ == "__main__": main()
