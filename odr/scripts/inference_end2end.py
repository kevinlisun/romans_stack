#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-01-09 15:10:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-21 17:52:26

import caffe
from odr.srv import *
from odr.msg import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc, ndimage
import cv2
import numpy as np
import time

def get_bboxes_imgs(image, bboxes):
	# image is the normailized image (np) W*H*C
	# bboxes is the bboxes array N*4, N is the number of bboxes
	print "shape of image:"
	print image.shape

	width = 224
	height = 224

	if len(image.shape) == 3:
		C = 3
	else:
		C = 1

	imgs = np.zeros( (bboxes.shape[0], C, width, height), dtype=np.float32 )

	for i in xrange(bboxes.shape[0]):

		min_x = bboxes[i,0]
		max_x = bboxes[i,1]
		min_y = bboxes[i,2]
		max_y = bboxes[i,3]

		edge_x = int(0.2*(max_x-min_x))
		edge_y = int(0.2*(max_y-min_y))

		center_shift = np.array([edge_x/2, edge_x/2, edge_y/2, edge_y/2])

		min_x = max(0, min_x-edge_x) + center_shift[0]
		max_x = min(image.shape[1], max_x+edge_x) + center_shift[1]
		min_y = max(0, min_y-edge_y) + center_shift[2]
		max_y = min(image.shape[0], max_y+edge_y) + center_shift[3]

		img = image[min_y:max_y, min_x:max_x]

		# resize the bbox image to the size of net input
		# img = misc.imresize( img, (width,height) )
		img = cv2.resize( img, (width,height) )

		if len(img.shape) < 3:
			img = img / 1000
			img = img[np.newaxis,...]
		else:
			# tanspose (W, H, C) to (C, H, W)
			img = img.transpose((2,0,1))

		imgs[i,:,:,:] = img

	return imgs


def infer(req):
	global caffe
	global net_rgb
	global net_depth
	global mode

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()

	bridge = CvBridge()

	# get rgb from message
	cv_rgb = bridge.imgmsg_to_cv2( req.rgb, desired_encoding="passthrough" )
	np_rgb = np.array( cv_rgb, dtype=np.float32 )

	# if the image is BGR for messages are already in RGB format (comment this)
	#np_rgb = np_rgb[:,:,::-1]
	# substract the mean
	np_rgb -= np.array((104.00698793,116.66876762,122.67891434))

	# get depth from messgae
	cv_depth = bridge.imgmsg_to_cv2( req.depth, desired_encoding="passthrough" )
	np_depth = np.array( cv_depth, dtype=np.float32 )

	np_depth -= np.array( 2 )

	# get bboxes from message
	cv_bboxes = bridge.imgmsg_to_cv2( req.bboxes, desired_encoding="passthrough" )

	np_bboxes = np.asarray( cv_bboxes, dtype=np.int )
	#print np_bboxes.shape

	# np_bboxes = np_bboxes[:,:,0]


	np_imgs_rgb = get_bboxes_imgs(np_rgb, np_bboxes)
	np_imgs_depth = get_bboxes_imgs(np_depth, np_bboxes)
	rospy.loginfo('receive proposal images:')
	#print np_imgs.shape

	# shape for input (data blob is N x C x H x W), set data
	# set the data layer prototxt
	net.blobs['rgb'].reshape(np_imgs_rgb.shape[0], np_imgs_rgb.shape[1], np_imgs_rgb.shape[2], np_imgs_rgb.shape[3])
	net.blobs['rgb'].data[...] = np_imgs_rgb
	net.blobs['depth'].reshape(np_imgs_depth.shape[0], np_imgs_depth.shape[1], np_imgs_depth.shape[2], np_imgs_depth.shape[3])
	net.blobs['depth'].data[...] = np_imgs_depth

	start_time = time.time()
	# forward propogation to extract features
	net.forward()
	rospy.loginfo('time consuming:')
	print(time.time()-start_time)

	# get the result blob
	np_confidence = net.blobs[req.layer.data].data

	'''np_confidence[:,0] = np_confidence[:,0] * 1.0
	np_confidence = np.exp(np_confidence)
	np_confidence_sum = np_confidence.sum(axis=1)
	np_confidence_sum = np_confidence_sum[...,np.newaxis]
	np_confidence = np_confidence / np.tile(np_confidence_sum,(1,11)) '''

	result = np_confidence.argmax(axis=1)

	rospy.loginfo('get results.')
	print result

	fea_msg = Feature()
	fea_msg.header = req.rgb.header
	fea_msg.data = result

	# print fea_msg

	res = InferenceResponse()
	res.result = fea_msg

	return res


def inference_server():
	
	rospy.init_node('end2end_inference_server')
	s = rospy.Service('/romans/end2end_inference', Inference, infer)

	print "Ready to call CNN Inference Service."
	rospy.spin()

if __name__ == "__main__":


	global caffe
	global net
	global mode

	print sys.argv[1]

	model_dirctory = sys.argv[1]

	mode = 'gpu'

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()
	

	# net = caffe.Net('/home/kevin/models/rgbd_net/romans_old/deploy.prototxt', '/home/kevin/models/rgbd_net/romans_old/romans_model.caffemodel', caffe.TEST)
	net = caffe.Net(model_dirctory + '/deploy.prototxt', model_dirctory + '/romans_model_fast.caffemodel', caffe.TEST)

	inference_server()
