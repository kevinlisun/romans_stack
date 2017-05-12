/*************************************************************************
	> File Name: normal_estimation_server.cpp
	> Author: Kevin Li Sunpointcloud_publisher.cpp
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/10/22
 ************************************************************************/


#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "odr/NormalEstimator.h"
#include "detector.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace message_filters;

ros::Publisher pub;

bool generateCloud(const sensor_msgs::ImageConstPtr rgb, const sensor_msgs::ImageConstPtr depth)
{
  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
  try
  {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb, sensor_msgs::image_encodings::BGR8);
    cv_depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }

  pcl::PCLPointCloud2 pc2;
  PointCloud::Ptr cloud (new PointCloud);

  CAMERA_INTRINSIC_PARAMETERS camera;
  camera = getDefaultCamera();

  std::vector<cv::Point2i> positions;
  positions.resize(0);

  image2PointCloud(cloud, cv_rgb_ptr->image, cv_depth_ptr->image, camera);

  // define the segmentation result (colored cloud)
  sensor_msgs::PointCloud2 colored_cloud_msg;

  // convert pcl::PointCloud to pcl::PointCloud2
  pcl::toPCLPointCloud2( *cloud, pc2 );

  // convert pcl::PointCLoud2 to sensor_msgs::PointCloud2
  pcl_conversions::fromPCL( pc2, colored_cloud_msg );
  //pcl::fromPCLPointCloud2( msg, pc );

  colored_cloud_msg.header = depth->header;

  pub.publish( colored_cloud_msg );

  return true;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "point_cloud_publisher");
  ros::NodeHandle n;

  message_filters::Subscriber<sensor_msgs::Image> sub_rgb(n, "/kinect2/qhd/image_color_rect", 1);
  message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, "/kinect2/qhd/image_depth_rect", 1);
  pub = n.advertise<sensor_msgs::PointCloud2> ("/romans/pointcloud_xyzrgb", 1);

  typedef sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_rgb, sub_depth);
  sync.registerCallback(boost::bind(&generateCloud, _1, _2));

  ROS_INFO("Ready to call point cloud publisher.");
  ros::spin();

  return 0;
}
