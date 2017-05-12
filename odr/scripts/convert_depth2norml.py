#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2016-11-09 16:18:24
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2016-12-12 01:01:10

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int8
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
from odr.srv import *
from odr.msg import *

import numpy as np
from scipy import misc, io
import time
import sys


def main():

	dataset_dir = '/home/kevin/dataset'

	index_file = '/home/kevin/dataset/processed_data/train.txt'

	read_stream = open(index_file, 'r')

	list = []

	for line in read_stream:
		list.append(line)


	file_Num = len(list)
	start_index = 228654
	end_index = 228664

	for i in range(start_index, end_index):

		print "extracting normal feature from " + str(i) + " of " + str(len(list)) + " file ..."

		tmp_str = list[i]

		info = tmp_str.split(" ")

		file = info[0]

		print '{}/processed_data/{}.mat'.format(dataset_dir, file)

		mat = io.loadmat('{}/processed_data/{}.mat'.format(dataset_dir, file))
		depth = mat['depth_map'].astype(np.float32)

		depth_msg = Image()
		bridge = CvBridge()
		depth_msg = bridge.cv2_to_imgmsg(depth, encoding='passthrough')

		NORMAL_ESTIMATION_SRV = '/romans/normal_estimator'

		try:
			rospy.loginfo("estimating normal ...")
			estimateNormal = rospy.ServiceProxy( NORMAL_ESTIMATION_SRV, NormalEstimator )
			n_res = estimateNormal( depth_msg )

			normal_map_msg = n_res.normal_map

			normal_map_mat = bridge.imgmsg_to_cv2( normal_map_msg, desired_encoding='passthrough' )
			normal_map_np = np.array(normal_map_mat, dtype=np.float32)

			# save the bbox images to the disk if it is required
			rospy.loginfo("save surface normal map to the disk ...")
			tmp = file.split("/")

			tmp_dir = ""
			for j in xrange(len(tmp)-1):
				tmp_dir = tmp_dir + "/" + tmp[j]

			file_j = tmp[len(tmp)-1]

			dir_j = '{}/normal_feature{}'.format(dataset_dir, tmp_dir)

			if not os.path.exists(dir_j):
				os.makedirs(dir_j)

			io.savemat('{}/normal_feature/{}'.format(dataset_dir, file), {'normal_map':np.array(normal_map_np)})
			print "done!"

		except rospy.ServiceException, e:
			print "Service call failed: %s"%e


if __name__ == '__main__':main()
