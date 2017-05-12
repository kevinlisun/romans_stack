#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "odr/FrameID.h"

#include <sstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{
  /**
   * The ros::init() function needs to see argc and argv so that it can perform
   * any ROS arguments and name remapping that were provided at the command line.
   * For programmatic remappings you can use a different version of init() which takes
   * remappings directly, but for most command-line programs, passing argc and argv is
   * the easiest way to do it.  The third argument to init() is the name of the node.
   *
   * You must call one of the versions of ros::init() before using any other
   * part of the ROS system.
   */
  ros::init(argc, argv, "kinect_simulator");

  ros::NodeHandle n;


  ros::Publisher rgb_pub = n.advertise<sensor_msgs::Image>("/kinect2/qhd/image_color_rect", 1);
  ros::Publisher depth_pub = n.advertise<sensor_msgs::Image>("/kinect2/qhd/image_depth_rect", 1);
  ros::Publisher fid_pub = n.advertise<odr::FrameID>("/kinect2/frame_id", 1);

  ros::Rate loop_rate(0.2);

  cv::Mat rgb, depth;
  // read image using cv::imread()


  int count = 0;

  while (ros::ok())
  {
    if (count >= 500)
      count = 0;

	  ostringstream int_str;
	  std::stringstream file_rgb, file_depth;

    file_rgb << argv[1] << "/rgb/" << count << ".jpg";
    file_depth << argv[1] << "/depth/" << count << ".png";

    std::cout << count << " " << file_rgb.str() << std::endl;
    std::cout << count << " " << file_depth.str() << std::endl;


    rgb = cv::imread( file_rgb.str() );
    depth = cv::imread( file_depth.str(), -1 );

    if (rgb.empty() || depth.empty())
    {
      std::cout << "skip ... imread(): image not found" << std::endl;
      count ++;
      continue;
    }

  	cv_bridge::CvImage cv_rgb, cv_depth;
    odr::FrameID fid_msg;

  	cv_rgb.encoding = sensor_msgs::image_encodings::BGR8;
    cv_depth.encoding = sensor_msgs::image_encodings::MONO16;

  	cv_rgb.image = rgb;
    cv_depth.image = depth;

    std_msgs::Header header;
    header.seq = 0;
    ros::Time now = ros::Time::now();
    header.stamp = now;
    header.frame_id = "/kinect2_rgb_optical_frame";

  	cv_rgb.header = header;
    cv_depth.header = header;

    fid_msg.header = header;
    fid_msg.data = count;

    // publish ros messages
    rgb_pub.publish( cv_rgb.toImageMsg() );
    depth_pub.publish( cv_depth.toImageMsg() );
    fid_pub.publish( fid_msg );

    ROS_INFO("Done!");
    std::cout << "frame " << count << " is published!" << std::endl;

    count ++;

    ros::spinOnce();

    loop_rate.sleep();
  }


  return 0;
}
