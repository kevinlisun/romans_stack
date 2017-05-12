/*************************************************************************
	> File Name: pcl_plus.h
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/09/15
 ************************************************************************/
# pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>

#include <pcl/features/don.h>
#include <pcl/io/pcd_io.h>


// define point cloud type
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void ppFiltering(PointCloud::Ptr & cloud);

void ppDownSampling(PointCloud::Ptr & cloud, double leaf_size);

void ppViewer(PointCloud::Ptr cloud);

void ppNormalEstimation( pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_p, pcl::PointCloud <pcl::Normal>::Ptr & normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree, double radius );

void ppRegSegmentation(PointCloud::Ptr cloud, std::vector <pcl::PointIndices> & clusters, pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree, bool is_visualization);

void ppColorRegSegmentation(PointCloud::Ptr cloud, std::vector <pcl::PointIndices> & clusters, bool is_visualization);

void ppDonSegmentation(PointCloud::Ptr cloud, bool is_visualization);

void ppGetInstanceMap(PointCloud::Ptr cloud, std::vector<pcl::PointIndices> clusters, cv::Mat& seg_map);

void ppGetColoredCloud(PointCloud::Ptr cloud, std::vector<pcl::PointIndices> clusters, PointCloud::Ptr & colored_cloud);



