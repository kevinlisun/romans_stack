/*
* @Author: Kevin Sun
* @Date:   2016-11-09 16:00:09
* @Last Modified by:   Kevin Sun
* @Last Modified time: 2016-12-12 01:01:36
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

  //convert sensor_msgs::PointCloud2 to pcl::PointCLoud2
  pcl_conversions::toPCL( req.pcl, pc2 );
  // convert pcl::PointCloud2 to pcl::PointCloud
  pcl::fromPCLPointCloud2( pc2, *cloud );
  
  Detector detector( cv_rgb_ptr->image, cv_depth_ptr->image, cloud );
  detector.estimateNormal( 2, 0.03 );
  detector.getClusters( 1 );
  detector.clusters2Bboxes();

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

  	colored_cloud_msg.header.frame_id = "/camera_rgb_optical_frame";

    res.proposals.colored_cloud = colored_cloud_msg;

  }
  
  int num_bboxes = detector.bboxes.size();

  cv::Mat mat(num_bboxes, 4, CV_32FC1);

  for( int i = 0; i < num_bboxes; i++ )
  {
    mat.at<float>(i,0) = detector.bboxes[i].min_x;
    mat.at<float>(i,1) = detector.bboxes[i].max_x;
    mat.at<float>(i,2) = detector.bboxes[i].min_y;
    mat.at<float>(i,3) = detector.bboxes[i].max_y;
  }

  cv_depth_ptr->image = mat;
  cv_depth_ptr->header = req.depth.header;
  res.proposals.bboxes = *(cv_depth_ptr->toImageMsg());

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

  ros::ServiceServer service = n.advertiseService("/romans/detector", detect);
  static ros::Publisher pub = n.advertise<sensor_msgs::PointCloud2> ("/romans/seg_result", 1);
  ROS_INFO("Ready to call detecor service.");
  ros::spin();

  return 0;
}
