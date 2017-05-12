/*************************************************************************
	> File Name: detector.h
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/09/16
 ************************************************************************/
# pragma once

#include "utility.h"

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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/features/don.h>
#include <pcl/io/pcd_io.h>

#include <std_msgs/Int8.h>

#include "cbf.h"

//#include <pcl/gpu/features/features.h>
//#include <pcl/gpu/containers/device_array.h>


// define point cloud type
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// define the structure of bounding box (four corners in image coordination system)
struct BoundingBox
{
	int min_x;
	int max_x;
	int min_y;
	int max_y;
};

inline std::ostream & operator << (std::ostream& stream, const BoundingBox& bbox)
{
  stream << bbox.min_x << " " <<  bbox.max_x << " " << bbox.min_y << " " << bbox.max_y;
  return stream;
}

// pcl based fuctions
void odrDownSampling ( PointCloud::Ptr & cloud, double leaf_size );
void odrNaNFiltering ( PointCloud::Ptr & cloud, double max_dist );
void odrViewer( PointCloud::Ptr colored_cloud );
void odrOutlinearRemoval ( PointCloud::Ptr & cloud, int num_neighbour, double thres );
void odrPlaneRemoval ( PointCloud::Ptr & cloud, double min_plane_size );
void odrFilterPlane( const PointCloud::Ptr & cloud, pcl::ModelCoefficients::Ptr & coefficients, pcl::PointIndices::Ptr & plane, double threshold );
void odrEstimateNormal ( pcl::PointCloud <pcl::PointXYZ>::Ptr cloud, pcl::PointCloud <pcl::Normal>::Ptr & normals, double radius );
void odrFillNaNValues( const cv::Mat & cv_rgb, const cv::Mat & cv_depth, cv::Mat & cv_refined_depth, const cv::Mat mask_, double scale );

class Detector
{
public:
	Detector()
	{
		camera = getDefaultCamera();

		cloud_p = pcl::PointCloud <pcl::PointXYZ>::Ptr (new pcl::PointCloud <pcl::PointXYZ>);
		copyPointCloud( *cloud, *cloud_p );

		// define searching method
		tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ> ());
		// initialize surface normal
		normals = pcl::PointCloud <pcl::Normal>::Ptr (new pcl::PointCloud <pcl::Normal>);
	}

	Detector( cv::Mat rgb_, cv::Mat depth_, float scale_ ) // here
	{
		scale = scale_;
		original_size = rgb_.size();

		if (scale == 1)
		{
			rgb = rgb_;
			depth = depth_;
		}
		else
		{
			cv::resize(rgb_, rgb, new_size, double(scale), double(scale));
			cv::resize(depth_, depth, new_size, double(scale), double(scale));
		}

		new_size = rgb.size();

		instance_map = cv::Mat(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0.));
		
		/*refineDepth( depth, refined_depth, 12 );
		std::cout << "here is okay" << std::endl;
		//medianFilter( refined_depth, 15, 15, CV_32F );
		
		for(int i = 0; i < 20; i++)
		{
			cv::Mat tmp;
			cv::medianBlur( refined_depth, tmp, 5 );
			refined_depth = tmp.clone();
			std::cout << "depth map refinement time: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
		}*/
		
		clock_t t = clock();
		camera = getDefaultCamera();

		cloud = PointCloud::Ptr (new PointCloud);
		image2PointCloud(cloud, rgb, depth, camera);
		std::cout << "pcl convesion time: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

		t = clock();
		odrNaNFiltering ( cloud, 2.0 );
		std::cout << "pre-processing time (NaN Removel): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

		t = clock();
		odrDownSampling( cloud, 0.01 ); // down-sample the cloud to 1CM, adjust this to accerlate the detection
		std::cout << "pre-processing time (Down Sampleing): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

		t = clock();
		odrPlaneRemoval ( cloud, 7000 ); // remove plane more than 3000 voxels, adjust this depending on your application
		std::cout << "pre-processing time (Plane Removel): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

		t = clock();
		odrOutlinearRemoval( cloud, 10, 0.05 );
		std::cout << "pre-processing time (Outlinear Removel): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

		cloud_p = pcl::PointCloud <pcl::PointXYZ>::Ptr (new pcl::PointCloud <pcl::PointXYZ>);
		copyPointCloud( *cloud, *cloud_p );

		// define searching method
		tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ> ());
		// initialize surface normal
		normals = pcl::PointCloud <pcl::Normal>::Ptr (new pcl::PointCloud <pcl::Normal>);
		cloud_with_normals = pcl::PointCloud <pcl::PointXYZINormal>::Ptr (new pcl::PointCloud<pcl::PointXYZINormal>);
	}

	Detector( cv::Mat rgb_, cv::Mat depth_, PointCloud::Ptr cloud_ )
	{	
		rgb = rgb_;
		depth = depth_;
		cloud = cloud_;
		instance_map = cv::Mat(depth.rows, depth.cols, CV_8UC1, cv::Scalar(0.));
		
		camera = getDefaultCamera();
		
		odrNaNFiltering ( cloud, 2.0 );
		odrDownSampling( cloud, 0.01 ); // down-sample the cloud to 1CM, adjust this to accerlate the detection
		odrPlaneRemoval ( cloud, 3000 );  // remove plane more than 3000 voxels, adjust this depending on your application
		odrOutlinearRemoval( cloud, 10, 0.05 );
		
		cloud_p = pcl::PointCloud <pcl::PointXYZ>::Ptr (new pcl::PointCloud <pcl::PointXYZ>);
		copyPointCloud( *cloud, *cloud_p );

		// define searching method
		tree = pcl::search::KdTree<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ> ());
		// initialize surface normal
		normals = pcl::PointCloud <pcl::Normal>::Ptr (new pcl::PointCloud <pcl::Normal>);
	}

	void estimateNormal(int mode, double radius);
	void getClusters(int method);
	void clusters2Bboxes();
	void refineDepth();
	void getInstanceMap();
	void getColoredCloud();
	
	
public:
	CAMERA_INTRINSIC_PARAMETERS camera;

	double scale;
	cv::Size original_size;
	cv::Size new_size;

	cv::Mat rgb;
	cv::Mat depth;
	cv::Mat instance_map;
	cv::Mat refined_depth;
	std::vector<std_msgs::Int8> instance_id;
	
	PointCloud::Ptr cloud, colored_cloud;
	pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_p; // cloud_p is XYZ format for computation

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;
	pcl::PointCloud <pcl::Normal>::Ptr normals;
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals;

	std::vector <pcl::PointIndices> clusters;
	std::vector<BoundingBox> bboxes;
};




