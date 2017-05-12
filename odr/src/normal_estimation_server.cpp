/*************************************************************************
	> File Name: normal_estimation_server.cpp
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/010/16
 ************************************************************************/


#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "odr/NormalEstimator.h"
#include "detector.h"
#include <sensor_msgs/Image.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


bool estimateNormal(odr::NormalEstimator::Request &req, odr::NormalEstimator::Response &res)
{
  cv_bridge::CvImagePtr cv_depth_ptr;
  cv_bridge::CvImage cv_normal;
  try
  {
    cv_depth_ptr = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC1);
    //cv_normal_ptr = cv_bridge::toCvCopy(req.depth, sensor_msgs::image_encodings::TYPE_32FC3);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }


  pcl::PointCloud <pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);

  CAMERA_INTRINSIC_PARAMETERS camera;
  camera = getVirtualCamera();

  std::vector<cv::Point2i> positions;
  positions.resize(0);

  image2PointCloud(cloud, positions, cv_depth_ptr->image, camera);

  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  //normals = pcl::PointCloud <pcl::Normal>::Ptr(new pcl::PointCloud <pcl::Normal>);
  odrEstimateNormal (cloud, normals, 0.05);

  int height = 424;
  int width = 512;

  cv::Mat normal_map(height, width, CV_32FC3, cv::Scalar(0.));

  normals2NormalMap( normal_map, normals, positions, height, width );

  cv_normal.image = normal_map;
  cv_normal.header = req.depth.header;
  cv_normal.encoding = sensor_msgs::image_encodings::TYPE_32FC3;
  res.normal_map = *(cv_normal.toImageMsg());

  //std::cout << res.normal_map << std::endl;

  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "normal_estimation_server");
  ros::NodeHandle n;

  ros::ServiceServer service = n.advertiseService("/romans/normal_estimator", estimateNormal);

  ROS_INFO("Ready to call normal_estimation service.");
  ros::spin();

  return 0;
}
