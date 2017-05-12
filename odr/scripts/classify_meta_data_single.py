#!/usr/bin/env python
import caffe
from svmutil import *

import numpy as np
from PIL import Image
from scipy import misc, io
import GPy

import os
import time
import sys
import shutil


def extract_feature( file ):
	global caffe 
	global net
	global mode

	width = 224
	height = 224

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	feature = np.array( [] )
	files = []


	print( 'extracting feature from ' + file + '...' )

	rgb = Image.open( file )
	np_rgb = np.array( rgb, dtype=np.float32 )

	if len(np_rgb.shape) is 2 or np_rgb.shape[2] is not 3:
		return np.array([])

	np_rgb = misc.imresize( np_rgb, (width,height) )

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

	feature = feat_fc7

	return feature


def gp_load_model(dirX='gp_model_X.npy', dirY='gp_model_Y.npy', dirParam='gp_model_param.npy'):
	# loading a model
	# Model creation, without initialization:
	X = np.load(dirX)
	Y = np.load(dirY)
	gp_model = GPy.models.GPClassification(X, Y, initialize=False)
	gp_model.update_model(False) # do not call the underlying expensive algebra on load
	gp_model.initialize_parameter() # Initialize the parameters (connect the parameters up)
	gp_model[:] = np.load('gp_model_param.npy') # Load the parameters
	gp_model.update_model(True) # Call the algebra only once
	print(gp_model)

	return gp_model


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
	threshold = 0.1

	mode = sys.argv[1]

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	net = caffe.Net('/home/kevin/models/vgg16/deploy.prototxt', '/home/kevin/models/vgg16/model.caffemodel', caffe.TEST)
	
	unlabelled_data_dir = '/home/kevin/catkin_ws/tmp'

	data_dir = '/home/kevin/catkin_ws/data/'
	folder = [ 'background_gp_labelled', 'cans_gp_labelled' ]
	label = [ 0, 1 ]

	for folderi in folder:
		if os.path.isdir(data_dir+folderi) == False:
			os.mkdir(data_dir+folderi)


	#------------------------------------GP labelling---------------------------------------------------------------------
	gp_model = gp_load_model()


	for file in os.listdir( unlabelled_data_dir ):

		if file.endswith( ".jpg" ) or file.endswith( ".png" ) or file.endswith( ".jpeg" ):

			instance = extract_feature( unlabelled_data_dir + "/" + file ) 

			if instance.size is 0:
				continue

			prob = gp_model.predict(instance)[0]
			print prob

			if prob > .5+threshold:
				src = unlabelled_data_dir + "/" + file
				dst = data_dir + "/" + folder[1] + "/" + file
				shutil.copy(src, dst)
				print "labelled as positive"
			elif prob < .5-threshold:
				src = unlabelled_data_dir + "/" + file
				dst = data_dir + "/" + folder[0] + "/" + file
				shutil.copy(src, dst)
				print "labelled as negetive"
			else:
				print "bad example"
	



if __name__ == "__main__": main()
