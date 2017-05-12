import colormap
import rospy
import numpy as np
import sys, os

from scipy import misc, io
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String, Float32

from odr.srv import *
from odr.msg import *


def main():

	rospy.init_node('detection_recognition', anonymous=True)

	VISUALIZATION_SRV = '/romans/visualization_server'
	SEMANTIC_CLOUD_PUB = '/romans/semantic_cloud'

	semantic_cloud_pub = rospy.Publisher( SEMANTIC_CLOUD_PUB, PointCloud2, queue_size=1 )

	bridge = CvBridge()

	file_name = sys.argv[1]

	mat = io.loadmat(os.path.join('/home/kevin/romans_bagfiles/evaluation/visualization', file_name +'_depth.mat'))
	depth = mat['depth'].astype(np.double)

	depth_msg = bridge.cv2_to_imgmsg( depth, encoding="passthrough" )


	mat = io.loadmat(os.path.join('/home/kevin/romans_bagfiles/evaluation/visualization', file_name +'_gt.mat'))
	semantic_map = mat['gt'].astype(np.double)

	semantic_map_msg = bridge.cv2_to_imgmsg( semantic_map, encoding="passthrough" )
	rospy.loginfo("visulizing semantic understanding ...")
	visualize = rospy.ServiceProxy( VISUALIZATION_SRV, Visualization )
	v_res = visualize( depth_msg, semantic_map_msg )
	# self.semantic_cloud_pub.publish( v_res.semantic_cloud )

	cv_img = bridge.imgmsg_to_cv2( v_res.colored_semantic_map, desired_encoding="passthrough" )
	np_img = np.asarray( cv_img )
	np_img = np_img[:,:,::-1] # if the image is bgr
	misc.imsave(os.path.join('/home/kevin/romans_bagfiles/evaluation/visualization', file_name +'gt.jpg'), np_img)
	print "result is saved!"


	mat = io.loadmat(os.path.join('/home/kevin/romans_bagfiles/evaluation/visualization', file_name +'_pd.mat'))
	semantic_map = mat['pd'].astype(np.double)

	semantic_map_msg = bridge.cv2_to_imgmsg( semantic_map, encoding="passthrough" )
	rospy.loginfo("visulizing semantic understanding ...")
	visualize = rospy.ServiceProxy( VISUALIZATION_SRV, Visualization )
	v_res = visualize( depth_msg, semantic_map_msg )
	semantic_cloud_pub.publish( v_res.semantic_cloud )

	bridge = CvBridge()
	cv_img = bridge.imgmsg_to_cv2( v_res.colored_semantic_map, desired_encoding="passthrough" )
	np_img = np.asarray( cv_img )
	np_img = np_img[:,:,::-1] # if the image is bgr
	misc.imsave(os.path.join('/home/kevin/romans_bagfiles/evaluation/visualization', file_name +'_pd.jpg'), np_img)
	print "result is saved!"

	rospy.spin()


if __name__ == '__main__': main()

	
	
