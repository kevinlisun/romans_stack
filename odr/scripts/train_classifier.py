#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-12-09 15:10:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-02-23 18:15:43

import caffe
from svmutil import *

import numpy as np
from PIL import Image
import cv2

from scipy import io

import GPy

import os
import time
import sys


def extract_feature( dir ):
	global caffe 
	global rgb_net
	global depth_net
	global mode
	global feat

	width = 224
	height = 224

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	feature = np.array( [] )
	files = []

	for file in os.listdir( dir ):

		if file.endswith( ".jpg" ) or file.endswith( ".png" ) or file.endswith( ".jpeg" ):
			print( 'extracting feature from ' + dir + '/' + os.path.splitext(file)[0] + '...' )

			if feat == 'rgb' or feat == 'rgbd':

				print "extracting rgb feature ..."
				
				rgb = Image.open( dir + '/' + file )
				# rgb = misc.imresize( rgb, (width,height) )
				rgb = cv2.resize( np.asarray(rgb), (width,height) )
				np_rgb = np.array( rgb, dtype=np.float32 )

				if len(np_rgb.shape) is 2 or np_rgb.shape[2] is not 3:
					continue

				# convert rgb to bgr
				np_rgb = np_rgb[:,:,::-1]
				# substract the mean
				np_rgb -= np.array( (104.00698793,116.66876762,122.67891434) )

				# transpose
				np_rgb = np_rgb.transpose( (2,0,1) )

				# shape for input (data blob is 1 x C x H x W), set data
				rgb_net.blobs['data'].reshape(1, *np_rgb.shape)
				rgb_net.blobs['data'].data[...] = np_rgb

				start_time = time.time()
				rgb_net.forward()
				print(time.time()-start_time)

				# get the feature blob
				rgb_feat_fc7 = np.array( rgb_net.blobs['fc7'].data, dtype=np.float32 )
				print rgb_feat_fc7.shape

			if feat == 'depth' or feat == 'rgbd':

				print "extracting depth feature ..."

				mat = io.loadmat('{}/{}.{}'.format(dir, os.path.splitext(file)[0], 'mat'))
				np_depth = mat['depth_map'].astype(np.double)

				# substract the mean
				np_depth -= np.array( 2 )

				# transpose
				np_depth = np_depth[np.newaxis, ...]

				# shape for input (data blob is 1 x C x H x W), set data
				depth_net.blobs['data'].reshape(1, *np_depth.shape)
				depth_net.blobs['data'].data[...] = np_depth

				start_time = time.time()
				depth_net.forward()
				print(time.time()-start_time)

				# get the feature blob
				depth_feat_fc7 = np.array( depth_net.blobs['fc7'].data, dtype=np.float32 )
				print depth_feat_fc7.shape


			if feat == 'rgb':
				feat_fc7 = rgb_feat_fc7
			elif feat == 'depth':
				feat_fc7 = depth_feat_fc7
			elif feat == "rgbd":
				feat_fc7 = np.concatenate( (rgb_feat_fc7, depth_feat_fc7), 1 )
			else:
				print "ERROR: unknown feature mode: " + feat

			if feature.size == 0:
				feature = feat_fc7
			else:
				feature = np.concatenate( (feature, feat_fc7), 0 )

	return feature

def gp_train(X, Y):

	opt = "rgbd"

	if opt == "rgbd":
		k_rgb = GPy.kern.RBF(X.shape[1]/2,variance=50,lengthscale=90, active_dims=np.arange(0,4096))
		k_depth = GPy.kern.RBF(X.shape[1]/2,variance=80,lengthscale=35, active_dims=np.arange(4096,8192))
		k = k_rgb * k_depth
	elif opt == "linear":
		k = GPy.kern.Linear(X.shape[1], active_dims=None, ARD=True)
	elif opt == "rbf":
		k = GPy.kern.RBF(X.shape[1], variance=50,lengthscale=90, active_dims=None)


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

	print k.K(X)
	#print k.variances
	io.savemat('./kernel.mat', {'K':np.array(k.K(X))})

	print m

	return m



def main():

	global caffe
	global rgb_net
	global depth_net
	global mode
	global feat

	#------------------------------------- setting ---------------------------------------------------------
	mode = sys.argv[1]
	feat = str(sys.argv[2])
	cat = str(sys.argv[3])

	classifier = "svm"

	print "-------------------------------------"
	print "caffe mode is: " + mode
	print "feature mode is: " + feat
	print "classifier is: " + classifier
	print "category is: " + cat
	print "-------------------------------------"

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	if feat == 'rgb' or feat == 'rgbd':
		rgb_net = caffe.Net('/home/kevin/models/vgg16/deploy.prototxt', '/home/kevin/models/vgg16/model.caffemodel', caffe.TEST)
	if feat == 'depth' or feat == 'rgbd':
		depth_net = caffe.Net('/home/kevin/models/model_net/heavy/deploy.prototxt', '/home/kevin/models/model_net/heavy/model.caffemodel', caffe.TEST)
	
	data_dir = '/home/kevin/catkin_ws/data/labelled/'
	folder = [ 'background', 'bottles', 'cans', 'chains', 'cloth', 'gloves', 'metal_objects', 'pipe_joints', 'plastic_pipes', 'sponges', 'wood_blocks' ]
	label = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

	
	#---------------------------------------------------------------------------------------------------------

	instances = np.array( [] )
	labels = np.array( [] )

	for foldi, labeli in zip( folder, label ):

		instance_i = extract_feature( os.path.join(data_dir, foldi) ) 

		if instances.size == 0:
			instances = instance_i
			labels = np.ones( [instance_i.shape[0]] ) * labeli
		else:
			instances = np.concatenate( ( instances, instance_i ), 0 )
			labels = np.concatenate( ( labels, np.ones( [instance_i.shape[0]]) * labeli ) )


	print instances.shape
	print labels.shape


	#-----------------------------------------train GP----------------------------------------------------------------
	if classifier == 'gp':
		
		labels = labels.reshape((labels.shape[0], 1))
		gp_model = gp_train(instances, labels)

		# save GP model to the disk
		X = instances
		Y = labels

		np.save('gp_model_param.npy', gp_model.param_array) 
		np.save('gp_model_X.npy', X)
		np.save('gp_model_Y.npy', Y)  

	#-----------------------------------------train SVM----------------------------------------------------------------
	if classifier == 'svm':

		model_file_name = './svm_model.model'
		io.savemat('./features.mat', {'instances':np.array(instances), 'labels':np.array(labels)})
		problem  = svm_problem( labels.tolist(), instances.tolist() )
		svm_opt = svm_parameter( '-t 0 -c 10' )
		model = svm_train( problem, svm_opt )
		svm_save_model( model_file_name, model )



if __name__ == "__main__": main()


