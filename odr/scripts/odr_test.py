#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-12-09 15:10:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-10 19:10:52

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from odr.srv import *
from odr.msg import *

# from svmutil import *
import numpy as np
from scipy import misc, io, ndimage
import time
import sys
import time

#import colormap

def save_bbox(img_msg, depth_msg, refined_depth_msg, inst_map_msg, bboxes_msg, instance_id):

	dir = '/home/kevin/catkin_ws/tmp/' # images directory
	if os.path.isdir(dir) == False:
		os.mkdir(dir)

	dir = '/home/kevin/catkin_ws/data/unlabelled/tmp' # patches (object proposals) directory
	if os.path.isdir(dir) == False:
		os.mkdir(dir)

	width = 224
	height = 224

	bridge = CvBridge()
	cv_img = bridge.imgmsg_to_cv2( img_msg, desired_encoding="passthrough" )
	cv_depth = bridge.imgmsg_to_cv2( depth_msg, desired_encoding="passthrough" )
	cv_refined_depth = bridge.imgmsg_to_cv2( refined_depth_msg, desired_encoding="passthrough" )
	cv_inst_map = bridge.imgmsg_to_cv2( inst_map_msg, desired_encoding="passthrough" )
	bboxes = bridge.imgmsg_to_cv2( bboxes_msg, desired_encoding="passthrough" )

	now = int(time.time())

	np_img = np.asarray( cv_img )
	np_img = np_img[:,:,::-1] # if the image is bgr
	io.savemat(os.path.join('/home/kevin/catkin_ws/tmp/', str(now)+'rgb.mat'), {'rgb':np.array(np_img)})


	np_depth = np.asarray( cv_depth, dtype=np.float32 )
	np_refined_depth = np.asarray( cv_refined_depth, dtype=np.float32 )
	np_inst_map = np.array( cv_inst_map, dtype=np.int8 )
	np_bboxes = np.asarray( bboxes, dtype=np.int )
	np_bboxes = np_bboxes[:,:]
	io.savemat(os.path.join('/home/kevin/catkin_ws/tmp/', str(now)+'_depth.mat'), {'depth_map':np.array(np_depth)})
	io.savemat(os.path.join('/home/kevin/catkin_ws/tmp/', str(now)+'_refined_depth.mat'), {'refined_depth_map':np.array(np_refined_depth)})
	io.savemat(os.path.join('/home/kevin/catkin_ws/tmp/', str(now)+'_instance_map.mat'), {'instance_map':np.array(np_inst_map)})


	print np_img.shape
	print np_depth.shape
	print np_bboxes.shape
	print instance_id

	for id_i, i in zip(instance_id, xrange(np_bboxes.shape[0])):
		tmp_depth = np.array(np_refined_depth)
		#tmp_depth[np_inst_map!=id_i.data] = 0 # uncomment here to mask the background

		min_x = np_bboxes[i,0]
		max_x = np_bboxes[i,1]
		min_y = np_bboxes[i,2]
		max_y = np_bboxes[i,3]

		edge_x = int(0.2*(max_x-min_x))
		edge_y = int(0.2*(max_y-min_y))

		center_shift = np.array([edge_x/2, edge_x/2, edge_y/2, edge_y/2])

		min_x = max(0, min_x-edge_x+center_shift[0])
		max_x = min(np_img.shape[1], max_x+edge_x+center_shift[1])
		min_y = max(0, min_y-edge_y+center_shift[2])
		max_y = min(np_img.shape[0], max_y+edge_y+center_shift[3])

		img = np_img[min_y:max_y, min_x:max_x]

		file_name = dir + "/" + "bbox_" + str(now) + "_" + str(i) + ".jpg"
		#img = misc.imresize( img, (width,height) )
		#print np.sum(np.nonzero(np.isnan(img).flatten()==True))
		img = cv2.resize( img, (width,height) )
		
		rospy.loginfo("save rgb image to the disk ...")
		misc.imsave(file_name, img)

		img = tmp_depth[min_y:max_y, min_x:max_x] / 1000
		#ks = int(max(min(img.shape[0],img.shape[1])/10,3))
		#img = ndimage.filters.median_filter(np.array(img,dtype=np.float32), ks)
		img = cv2.resize( img, (width,height) )

		file_name = dir + "/" + "bbox_" + str(now) + "_" + str(i) + ".mat"
		rospy.loginfo("save depth map to the disk ...")
		io.savemat(file_name, {'depth_map':np.array(img)})


class Detection_Recognition:


	def __init__(self):

		print sys.argv[1]

		if sys.argv[1] == 'kinect1':
			RGB_TOPIC =  '/camera/rgb/image_rect_color'
			DEPTH_TOPIC = '/camera/depth/image_rect'
			PCL_TOPIC = '/camera/depth_registered/points'
			SEG_RESULT_PUB = '/romans/seg_result'

		elif sys.argv[1] == 'kinect2':
			# define topics
			RGB_TOPIC = '/kinect2/qhd/image_color_rect' 
			DEPTH_TOPIC = '/kinect2/qhd/image_depth_rect'
			CAMERA_INFO_TOPIC = 'kinect2/qhd/camera_info'
			PCL_TOPIC = '/kinect2/qhd/points'
			SEG_RESULT_PUB = '/romans/seg_result'
			SEMANTIC_MAP_PUB = '/romans/semantic_map'
			SEMANTIC_CLOUD_PUB = '/romans/semantic_cloud'
			RGB_TOPIC_PUB = '/romans/rgb' 
			DEPTH_TOPIC_PUB = '/romans/depth'
			CAMERA_INFO_PUB = '/romans/camera'


		rospy.init_node('detection_recognition', anonymous=True)

		# define publisher, subscriber
		self.rgb_sub = message_filters.Subscriber( RGB_TOPIC, Image, queue_size=1, buff_size=2**24 )
		self.depth_sub = message_filters.Subscriber( DEPTH_TOPIC, Image, queue_size=1, buff_size=2**24 )
		self.camera_info_sub = message_filters.Subscriber( CAMERA_INFO_TOPIC, CameraInfo, queue_size=1 )
		# self.pcl_sub = rospy.Subscriber( PCL_TOPIC, PointCloud2, self.getCloud, queue_size=1 )
		ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.camera_info_sub], 1, 1)
		ts.registerCallback(self.doDectionAndRecognition)

		self.rgb_pub = rospy.Publisher( RGB_TOPIC_PUB, Image, queue_size=1 )
		self.depth_pub = rospy.Publisher( DEPTH_TOPIC_PUB, Image, queue_size=1 )
		self.seg_pub = rospy.Publisher( SEG_RESULT_PUB, PointCloud2, queue_size=1 )
		self.semantic_map_pub = rospy.Publisher( SEMANTIC_MAP_PUB, Image, queue_size=1 )
		self.semantic_cloud_pub = rospy.Publisher( SEMANTIC_CLOUD_PUB, PointCloud2, queue_size=1 )
		self.camera_info_pub = rospy.Publisher( CAMERA_INFO_PUB, CameraInfo, queue_size=1 )

		rospy.spin()


	def doDectionAndRecognition( self, rgb, depth, camera ):

		# define service clients
		DETECTOR_SRV = '/romans/detector'
		FEATURE_EXTRACTOR_SRV = '/romans/feature_extractor'
		NORMAL_ESTIMATION_SRV = '/romans/normal_estimator'
		VISUALIZATION_SRV = '/romans/visualization_server'
		INFERENCE_SRV = '/romans/end2end_inference'

		SCALE = 1.0

		# detection, get object proposals
		rospy.wait_for_service('/romans/detector')
	
		try:
			rospy.loginfo("detecting object bboxes ...")
			detect = rospy.ServiceProxy( DETECTOR_SRV, Detector )
			pcl = PointCloud2()

			d_res = detect( rgb, depth, pcl, Float32(SCALE) )

			instance_id = d_res.proposals.instance_id
		
			# save the bbox images to the disk if it is required
			rospy.loginfo("save proposal images to the disk ...")
			start_time = time.time()
			# save_bbox(rgb, depth, d_res.masked_depth, d_res.proposals.instance_map, d_res.proposals.bboxes, instance_id)
			print "saving images runing time: " + str(time.time()-start_time)
			print "done!"

			# pushlish segmentation result (colored_cloud)
			self.seg_pub.publish(d_res.proposals.colored_cloud)
			rospy.loginfo("detection result is published!")


		except rospy.ServiceException, e:
			print "Service call failed: %s"%e


		try:

			rospy.loginfo("end-to-end inference ...")
			infer = rospy.ServiceProxy( INFERENCE_SRV, Inference )
			r_res = infer( rgb, d_res.masked_depth, d_res.proposals.bboxes, String('rgbd_fc8') )

			p_label = r_res.result.data

			bridge = CvBridge()
			cv_mat = bridge.imgmsg_to_cv2( d_res.proposals.instance_map, desired_encoding="passthrough" )
			instance_map = np.array( cv_mat, dtype=np.int8 )
			semantic_map = np.zeros_like( instance_map, dtype=np.uint8 )

			for id_i, p_i in zip(instance_id, p_label):
				semantic_map[instance_map==id_i.data] = p_i


			start_time = time.time()
			semantic_map_msg = bridge.cv2_to_imgmsg( semantic_map, encoding="passthrough" )
			rospy.loginfo("visulizing semantic understanding ...")
			visualize = rospy.ServiceProxy( VISUALIZATION_SRV, Visualization )
			v_res = visualize( depth, semantic_map_msg )

			self.semantic_cloud_pub.publish( v_res.semantic_cloud )

			print "Colormap display running time: " + str(time.time()-start_time)

			stamp_now = rospy.Time.now() 
			rgb.header.stamp = stamp_now
			depth.header.stamp = stamp_now
			camera.header.stamp = stamp_now
			v_res.colored_semantic_map.header.stamp = stamp_now

			self.rgb_pub.publish( rgb )
			self.depth_pub.publish( depth )
			self.camera_info_pub.publish( camera )
			self.semantic_map_pub.publish( v_res.colored_semantic_map )
			
			print "result is published!"


		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		

    

if __name__ == '__main__':
	dr = Detection_Recognition()
