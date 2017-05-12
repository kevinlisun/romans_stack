/*
* @Author: Kevin Sun
* @Date:   2016-12-09 15:10:13
* @Last Modified by:   Kevin Sun
* @Last Modified time: 2017-05-11 19:45:25
*/
#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "odr/Detector.h"
#include "detector.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


bool detect(odr::Detector::Request  &req, odr::Detector::Response &res)
{
  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr, cv_bbox_ptr;
  try
  {
    cv_rgb_ptr = cv_bridge::toCvCopy(req.rgb, sensor_msgs::image_encodings::BGR8);
    cv_depth_ptr = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  // define pcl::PointCloud2 for conversion
  pcl::PCLPointCloud2 pc2;
  PointCloud::Ptr cloud (new PointCloud());

  // convert sensor_msgs::PointCloud2 to pcl::PointCLoud2
  //pcl_conversions::toPCL( req.pcl, pc2 );
  // convert pcl::PointCloud2 to pcl::PointCloud
  //pcl::fromPCLPointCloud2( pc2, *cloud );

  float scale = req.scale.data;

  Detector detector( cv_rgb_ptr->image, cv_depth_ptr->image, scale );
  detector.estimateNormal( 3, 0.03 ); // 1 refers to GPU, 2 refers to single-thread, 3 refers to multi-thread
  detector.getClusters( 3 ); // 1 refers to "Region Growing", 2 refers to "Euclidean Clustering", 3 refers to "Conditional Clustering"
  detector.clusters2Bboxes();
  detector.refineDepth();
  detector.getInstanceMap();

  bool publish_result = true;

  if (publish_result)
  {
    // define the segmentation result (colored cloud)
    sensor_msgs::PointCloud2 colored_cloud_msg;

  	detector.getColoredCloud();
    ROS_INFO("get segmettation result, colored_cloud:");	
  	std::cout << *(detector.colored_cloud) << std::endl;

  	// convert pcl::PointCloud to pcl::PointCloud2
  	pcl::toPCLPointCloud2( *(detector.colored_cloud), pc2 );

  	// convert pcl::PointCLoud2 to sensor_msgs::PointCloud2
  	pcl_conversions::fromPCL( pc2, colored_cloud_msg );
  	//pcl::fromPCLPointCloud2( msg, pc );

  	colored_cloud_msg.header = req.depth.header;

    res.proposals.colored_cloud = colored_cloud_msg;

  }

  cv::Mat mask = detector.instance_map >= 0;
  cv::Mat masked_depth;
  (detector.refined_depth).copyTo(masked_depth, mask);
  cv_depth_ptr->image = masked_depth;
  cv_depth_ptr->header = req.depth.header;
  res.masked_depth = *(cv_depth_ptr->toImageMsg());
  
  int num_bboxes = detector.bboxes.size();

  cv::Mat mat(num_bboxes, 4, CV_32FC1);

  for( int i = 0; i < num_bboxes; i++ )
  {
    mat.at<float>(i,0) = detector.bboxes[i].min_x / scale;
    mat.at<float>(i,1) = detector.bboxes[i].max_x / scale;
    mat.at<float>(i,2) = detector.bboxes[i].min_y / scale;
    mat.at<float>(i,3) = detector.bboxes[i].max_y / scale;
  }

  cv_depth_ptr->image = mat;
  cv_depth_ptr->header = req.depth.header;
  res.proposals.bboxes = *(cv_depth_ptr->toImageMsg());

  cv_depth_ptr->image = detector.instance_map;
  cv_depth_ptr->encoding = sensor_msgs::image_encodings::TYPE_8UC1;
  res.proposals.instance_map = *(cv_depth_ptr->toImageMsg());

  std::vector<std_msgs::Int8> v = detector.instance_id;
  res.proposals.instance_id = v;

  //cv_bridge::CvImage cv_msg;
  //cv_msg.header = req.depth.header;
  //cv_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;;
  //cv_msg.image = mat;

  //sensor_msgs::Image msg = *(cv_msg.toImageMsg());
  //res.proposals.bboxes = msg;

  //std::cout << msg << std::endl;

  ROS_INFO( "%d bounding boxes are generated.", num_bboxes );
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "detection_server");
  ros::NodeHandle n;

  /*int device = argc > 1 ? atoi(argv[1]) : 0;
  af::setDevice(device);
  af::info();*/

  ros::ServiceServer service = n.advertiseService("/romans/detector", detect);
  ROS_INFO("Ready to call detecor service.");
  ros::spin();

  return 0;
}
