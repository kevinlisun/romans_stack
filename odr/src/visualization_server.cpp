/*
* @Author: Kevin Sun
* @Date:   2016-11-18 14:36:53
* @Last Modified by:   Kevin Sun
* @Last Modified time: 2016-11-18 18:49:56
*/


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include "detector.h"
#include "visualization.h"
#include "odr/Visualization.h"

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

CAMERA_INTRINSIC_PARAMETERS camera;

bool visualize(odr::Visualization::Request &req, odr::Visualization::Response &res)
{

  cv_bridge::CvImagePtr cv_depth_ptr;
  cv_bridge::CvImagePtr cv_smap_ptr;
  cv_bridge::CvImage cv_cmap;

  try
  {
    cv_depth_ptr = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1);
    cv_smap_ptr = cv_bridge::toCvCopy(req.semantic_map, sensor_msgs::image_encodings::TYPE_8UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  //std::cout << cv_smap_ptr->image << std::endl;
  // get colored semantic map
  cv::Mat semantic_map = cv_smap_ptr->image;
  cv::Mat colored_semantic_map = getColoredSemanticMap( semantic_map );

  //std::cout << colored_semantic_map << std::endl;

  cv_smap_ptr->image = colored_semantic_map;
  cv_smap_ptr->header = req.depth.header;
  cv_smap_ptr->encoding = sensor_msgs::image_encodings::TYPE_8UC3;
  res.colored_semantic_map = *(cv_smap_ptr->toImageMsg()); // return colored semantic map

  // get colored semantic cloud
  PointCloud::Ptr semantic_cloud (new PointCloud);
  image2PointCloud(semantic_cloud, colored_semantic_map, cv_depth_ptr->image, camera);
  //convert point cloud to message and return
  pcl::PCLPointCloud2 pc2;
  
  sensor_msgs::PointCloud2 semantic_cloud_msg; // define the semantic cloud result (colored_semantic_cloud)
  pcl::toPCLPointCloud2( *semantic_cloud, pc2 ); // convert pcl::PointCloud to pcl::PointCloud2
  pcl_conversions::fromPCL( pc2, semantic_cloud_msg ); // convert pcl::PointCLoud2 to sensor_msgs::PointCloud2

  semantic_cloud_msg.header = req.depth.header;

  res.semantic_cloud = semantic_cloud_msg;
  
  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "visualization_server");
  ros::NodeHandle n;

  camera = getDefaultCamera();

  ros::ServiceServer service = n.advertiseService("/romans/visualization_server", visualize);

  ROS_INFO("Ready to call visualization service.");
  ros::spin();

  return 0;
}
