#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-12-09 15:10:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-02-23 16:31:42

import caffe
from odr.srv import *
from odr.msg import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc
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

		edge_x = int(0.25*(max_x-min_x))
		edge_y = int(0.25*(max_y-min_y))

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
			img = img[np.newaxis,...]
		else:
			# tanspose (W, H, C) to (C, H, W)
			img = img.transpose((2,0,1))

		imgs[i,:,:,:] = img

	return imgs


def extract_feature(req):
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

	if req.option.data == 'rgb_only' or req.option.data == 'rgb_depth':

		cv_rgb = bridge.imgmsg_to_cv2( req.rgb, desired_encoding="passthrough" )
		np_rgb = np.array( cv_rgb, dtype=np.float32 )

		# if the image is bgr
		np_rgb = np_rgb[:,:,::-1]
		# substract the mean
		np_rgb -= np.array((104.00698793,116.66876762,122.67891434))

	if req.option.data == 'depth_only' or req.option.data == 'rgb_depth':
		
		cv_depth = bridge.imgmsg_to_cv2( req.depth, desired_encoding="passthrough" )
		np_depth = np.array( cv_depth, dtype=np.float32 )

		np_depth -= np.array( 2 )



	# get bboxes from message
	cv_bboxes = bridge.imgmsg_to_cv2( req.bboxes, desired_encoding="passthrough" )

	np_bboxes = np.asarray( cv_bboxes, dtype=np.int )
	#print np_bboxes.shape

	# np_bboxes = np_bboxes[:,:,0]

	print req.option
	# get bounding box images
	if req.option.data == 'rgb_only' or req.option.data == 'rgb_depth':
		np_imgs_rgb = get_bboxes_imgs(np_rgb, np_bboxes)
		rospy.loginfo('receive proposal images:')
		#print np_imgs.shape

		# shape for input (data blob is N x C x H x W), set data
		# set the data layer prototxt
		net_rgb.blobs['data'].reshape(np_imgs_rgb.shape[0], np_imgs_rgb.shape[1], np_imgs_rgb.shape[2], np_imgs_rgb.shape[3])
		net_rgb.blobs['data'].data[...] = np_imgs_rgb

		start_time = time.time()
		# forward propogation to extract features
		net_rgb.forward()
		rospy.loginfo('time consuming:')
		print(time.time()-start_time)

		# get the feature blob
		feat_rgb_fc7 = np.asarray( net_rgb.blobs['fc7'].data, dtype=np.float32 )

		rospy.loginfo('get rgb features:')
		print feat_rgb_fc7.shape

	if req.option.data == 'depth_only' or req.option.data == 'rgb_depth':
		
		np_imgs_depth = get_bboxes_imgs(np_depth, np_bboxes)
		rospy.loginfo('receive proposal images:')
		print np_imgs_depth.shape

		# shape for input (data blob is N x C x H x W), set data
		# set the data layer prototxt
		net_depth.blobs['data'].reshape(np_imgs_depth.shape[0], np_imgs_depth.shape[1], np_imgs_depth.shape[2], np_imgs_depth.shape[3])
		net_depth.blobs['data'].data[...] = np_imgs_depth

		start_time = time.time()
		# forward propogation to extract features
		net_depth.forward()
		rospy.loginfo('time consuming:')
		print(time.time()-start_time)

		# get the feature blob
		feat_depth_fc7 = np.asarray( net_depth.blobs['fc7'].data, dtype=np.float32 )

		rospy.loginfo('get depth features:')
		print feat_depth_fc7.shape

	if req.option.data == 'rgb_only':
		feat_fc7 = feat_rgb_fc7
	elif req.option.data == 'depth_only':
		feat_fc7 = feat_depth_fc7
	elif req.option.data == 'rgb_depth':
		feat_fc7 = np.concatenate( (feat_rgb_fc7, feat_depth_fc7), 1 )

	fea_msg = Feature()
	fea_msg.header = req.rgb.header
	fea_msg.width = feat_fc7.shape[1]
	fea_msg.height = feat_fc7.shape[0]

	# flatten features to 1d vector
	feat_fc7 = feat_fc7.flatten()
	fea_msg.data = feat_fc7.tolist()

	# print fea_msg

	res = FeatureExtractorResponse()
	res.features = fea_msg

	return res


def feature_extraction_server():
	
	rospy.init_node('feature_extraction_server')
	s = rospy.Service('/romans/feature_extractor', FeatureExtractor, extract_feature)

	print "Ready to call Feature Extraction Service."
	rospy.spin()

if __name__ == "__main__":


	global caffe
	global net_rgb
	global net_depth
	global mode

	print sys.argv[1]

	mode = sys.argv[1]

	if mode == 'cpu':
		caffe.set_mode_cpu()
	else:
		caffe.set_device(0)
		caffe.set_mode_gpu()
	

	net_rgb = caffe.Net('/home/kevin/models/vgg16/deploy.prototxt', '/home/kevin/models/vgg16/model.caffemodel', caffe.TEST)
	net_depth = caffe.Net('/home/kevin/models/model_net/heavy/deploy.prototxt', '/home/kevin/models/model_net/heavy/model.caffemodel', caffe.TEST)

	feature_extraction_server()
