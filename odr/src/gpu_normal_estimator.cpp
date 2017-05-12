/*************************************************************************
	> File Name: gpu_normal_estimator.cpp
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/09/16
 ************************************************************************/

# pragma once

#include "utility.h"
#include <pcl/gpu/features.h>


int computeGPUNormal()
{

	pcl::gpu::Feature::PointCloud cloud_d(cloud_->width * cloud_->height); 
    cloud_d.upload(cloud_->points); 
    pcl::gpu::NormalEstimation ne_d; 
    ne_d.setInputCloud(cloud_d); 
    ne_d.setViewPoint(0, 0, 0); 
    ne_d.setRadiusSearch(0.01, 4); 
    pcl::gpu::Feature::Normals normals_d(cloud_->width * cloud_->height); 

    ne_d.compute(normals_d);     
}

int main(int arc, char ** argv)
{
	ros::init(argc, argv, "gpu_normal_estimator");
	ros::NodeHandle n;

	ros::Subscriber sub = n.subscriber("/romans/gpu_normal_estimator", 1, computeGPUNormal);
}

