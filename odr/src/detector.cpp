/*************************************************************************
	> File Name: detector.cp
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/09/16
 ************************************************************************/

#include "detector.h"

#define getMax( x, y ) ( (x) > (y) ? (x) : (y) )
#define getMin( x, y ) ( (x) < (y) ? (x) : (y) )


void odrViewer( PointCloud::Ptr colored_cloud )
{
    pcl::visualization::CloudViewer viewer ("Cluster viewer");
    viewer.showCloud(colored_cloud);
    while (!viewer.wasStopped ())
    {
    }
}
void odrDownSampling ( PointCloud::Ptr & cloud, double leaf_size )
{

    pcl::PCLPointCloud2::Ptr cloud_p ( new pcl::PCLPointCloud2() );
    pcl::toPCLPointCloud2( *cloud, *cloud_p );
    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud ( cloud_p );
    sor.setLeafSize ( leaf_size, leaf_size, leaf_size );
    sor.filter ( *cloud_p );

    std::cerr << "PointCloud after filtering: " << cloud_p->width * cloud_p->height 
       << " data points (" << pcl::getFieldsList (*cloud_p) << ").";
    std::cout << std::endl;

    pcl::fromPCLPointCloud2( *cloud_p, *cloud );
}

void odrNaNFiltering ( PointCloud::Ptr & cloud, double max_dist )
{
    // remove NaN values
    pcl::IndicesPtr indices ( new std::vector <int> );
    pcl::removeNaNFromPointCloud( *cloud, *cloud, *indices );

    pcl::PassThrough<PointT> pass;
    pass.setInputCloud ( cloud );
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, max_dist);
    pass.filter (*cloud);
}

void odrOutlinearRemoval ( PointCloud::Ptr & cloud, int num_neighbour, double thres )
{

	//cloud_p = pcl::PointCloud <pcl::PointXYZ>::Ptr (new pcl::PointCloud <pcl::PointXYZ>);
	//copyPointCloud( *cloud, *cloud_p );
	// Create the filtering object
  	pcl::StatisticalOutlierRemoval<PointT> sor;
  	sor.setInputCloud ( cloud ) ;
  	sor.setMeanK ( num_neighbour );
  	sor.setStddevMulThresh ( thres );
  	sor.filter ( *cloud );
}

void odrFilterPlane( const PointCloud::Ptr & cloud, pcl::ModelCoefficients::Ptr & coefficients, pcl::PointIndices::Ptr & plane, double threshold )
{
    (*plane).indices.resize(0);
    double a = (*coefficients).values[0];
    double b = (*coefficients).values[1];
    double c = (*coefficients).values[2];
    double d = (*coefficients).values[3];

    // std::cout <<" a:" << a << " b:" << b << " c:" << c << " d:" << d << std::endl;

    for (int i = 0; i < (cloud->points).size(); i++ )
    {
      if ( std::abs( a*cloud->points[i].x + b*cloud->points[i].y + c*cloud->points[i].z + d ) < threshold )
      {
        (*plane).indices.push_back(i);
      }
    }

}


void odrPlaneRemoval ( PointCloud::Ptr & cloud, double min_plane_size )
{
    PointCloud::Ptr tmp_cloud (new PointCloud);
    copyPointCloud( *cloud, *tmp_cloud );
    odrDownSampling( tmp_cloud, 0.02 );

	  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  	pcl::PointIndices::Ptr plane (new pcl::PointIndices);
    pcl::PointIndices::Ptr tmp_plane (new pcl::PointIndices);

  	// Create the segmentation object
  	pcl::SACSegmentation<PointT> seg;
  	// Optional
  	seg.setOptimizeCoefficients (false);
  	// Mandatory
  	seg.setModelType (pcl::SACMODEL_PLANE);
  	seg.setMethodType (pcl::SAC_RANSAC);
  	seg.setMaxIterations (700);
  	seg.setDistanceThreshold (0.02);

  	// Create the filtering object
  	pcl::ExtractIndices<PointT> extract;

  	bool done = false;

  	while (!done)
  	{
  		seg.setInputCloud (tmp_cloud);
  		seg.segment (*tmp_plane, *coefficients);
      odrFilterPlane( cloud, coefficients, plane, 0.02 );

  		if (plane->indices.size() < min_plane_size)
  		{
  			done = true;
  		}
  		else
  		{
        //std::cout << "cloud: " << cloud->points.size() << std::endl;
        //std::cout << "plane: " << (*plane).indices.size() << std::endl;
  			extract.setInputCloud (cloud);
    		extract.setIndices (plane);
    		extract.setNegative (true);
    		extract.filter (*cloud);

        extract.setInputCloud (tmp_cloud);
        extract.setIndices (tmp_plane);
        extract.setNegative (true);
        extract.filter (*tmp_cloud);
    	}
  	}
}

void odrEstimateNormal ( pcl::PointCloud <pcl::PointXYZ>::Ptr cloud, pcl::PointCloud <pcl::Normal>::Ptr & normals, double radius )
{
	clock_t t = clock();
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setNumberOfThreads( 8 );
	normal_estimator.setSearchMethod ( tree );
	normal_estimator.setInputCloud ( cloud );
	normal_estimator.setRadiusSearch ( radius );
	// normal_estimator.setKSearch (50);
	normal_estimator.compute ( *normals );

	std::cout << "normal estimation time (multi-thresd): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
}

void odrFillNaNValues( const cv::Mat & rgb_, const cv::Mat & depth_, cv::Mat & refined_depth, const cv::Mat mask_, double scale )
{

  cv::Size new_size;

  // resize images
  cv::Mat cv_rgb, cv_depth, cv_mask, cv_depth_normed;

  if(scale == 1)
  {
    cv_rgb = rgb_.clone();
    cv_depth = depth_.clone();
    cv_mask = mask_.clone();
  }
  else
  {
    cv::resize(rgb_, cv_rgb, new_size, scale, scale);
    cv::resize(depth_, cv_depth, new_size, scale, scale);
    cv::resize(mask_, cv_mask, new_size, scale, scale);
  }

  int height = cv_depth.rows;
  int width = cv_depth.cols;

  std::cout << height << " " << width << std::endl;

  // convert rgb to intensity image
  cv::Mat cv_intensity(cv_rgb.size(), CV_8UC1);
  cvtColor( cv_rgb, cv_intensity, CV_BGR2GRAY );

  // normalize depth to [0 1] maxium is 2
  //depth_.convertTo(cv_depth, CV_32FC1, 2.0);
  double min, max;
  //cv::minMaxLoc(cv_depth, &min, &max);
  min = 0;
  max = 2.0 * 1000;
  cv_depth = (cv_depth - cv::Scalar(min)) * (1/max);
  std::cout << min << " " << max << std::endl;
  cv_depth.convertTo(cv_depth_normed, CV_8UC1, 255.0);

  // create mask >= 2.0 meter
  cv::Mat nan_mask(cv_depth.size(), CV_8UC1);
  cv::bitwise_and(cv_depth <= 2.0, cv_depth == 0, nan_mask);
  cv::bitwise_and(cv_mask, nan_mask, cv_mask);
  

  cv::Mat cv_result_normed(cv_depth.size(), CV_8UC1);

  unsigned num_scales = 3;
  double sigma_s[] = {12, 5, 8};
  double sigma_r[] = {0.2, 0.08, 0.02};

  int N = height * width;
  uint8_t* depth = (uint8_t*) malloc(N * sizeof(uint8_t));
  uint8_t* intensity = (uint8_t*) malloc(N * sizeof(uint8_t));
  uint8_t* result = (uint8_t*) malloc(N * sizeof(uint8_t));
  bool* mask = (bool*) malloc(N * sizeof(bool));

  for( int i = 0; i < height; i++ )
    for( int j = 0; j < width; j++ )
    {
      *(depth + j*height + i) = uint8_t(cv_depth_normed.ptr<uchar>(i)[j]);
      *(intensity + j*height + i) = uint8_t(cv_intensity.ptr<uchar>(i)[j]);
      *(mask + j*height + i) = cv_mask.ptr<uchar>(i)[j]==255.0 ;
    }

  cbf::cbf( height, width, depth, intensity, mask, result, num_scales, sigma_s, sigma_r );

  for( int i = 0; i < height; i++ )
    for( int j = 0; j < width; j++ )
    {
      cv_result_normed.at<uchar>(i,j) = uchar(*(result + j*height + i));
    }

  //cv::imwrite("reuslt.jpg", cv_result_normed);

  free(depth);
  free(intensity);
  free(result);
  free(mask);

  cv::Mat cv_result;
  cv_result_normed.convertTo(cv_result, CV_32FC1, 1/255.0);
  cv_result = cv_result * max + cv::Scalar(min);
  cv::resize(cv_result, refined_depth, cv::Size(depth_.cols, depth_.rows));

  bool is_visualize = false;

  if( is_visualize )
  {
    char window_name[] = "cv_result_normed";
    cv::namedWindow( window_name, CV_WINDOW_NORMAL );
    cv::resizeWindow( window_name, 400, 300 );
    //img.copyTo( dst, edge );
    cv::imshow( window_name, cv_result_normed );

    // Wait until user exit program by pressing a key
    cv::waitKey(500);
  }
  
}

bool enforceIntensitySimilarity (const pcl::PointXYZINormal & point_a, const pcl::PointXYZINormal & point_b, float squared_distance)
{
  if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  else
    return (false);
}

bool enforceCurvatureOrIntensitySimilarity (const pcl::PointXYZINormal & point_a, const pcl::PointXYZINormal & point_b, float squared_distance)
{
// ROS indigo  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.normal, point_b_normal = point_b.normal;
// ROS kinectic
  const Eigen::Vector3f point_a_normal(point_a.normal[0], point_a.normal[1], point_a.normal[2]);
  const Eigen::Vector3f point_b_normal(point_b.normal[0], point_b.normal[1], point_b.normal[2]);
  if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
    return (true);
  return (false);
}

bool customRegionGrowing (const pcl::PointXYZINormal & point_a, const pcl::PointXYZINormal & point_b, float squared_distance)
{
// ROS indigo
//  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.normal, point_b_normal = point_b.normal;
// ROS kinectic
  const Eigen::Vector3f point_a_normal(point_a.normal[0], point_a.normal[1], point_a.normal[2]);
  const Eigen::Vector3f point_b_normal(point_b.normal[0], point_b.normal[1], point_b.normal[2]);
  //std::cout << squared_distance << std::endl;
  if (squared_distance <= 0.01)
  {
    if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
      return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
      return (true);
  }
  else
  {
    if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
      return (true);
  }
  return (false);
}


void Detector::estimateNormal( int mode, double radius )
{
	if ( mode == 1 ) // cpu, single thred mode
	{
		  clock_t t = clock();
    	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    	normal_estimator.setSearchMethod ( tree );
    	normal_estimator.setInputCloud ( cloud_p );
    	normal_estimator.setRadiusSearch ( radius );
    	// normal_estimator.setKSearch (50);
    	normal_estimator.compute ( *normals );
    	std::cout << "normal estimation time (single-thresd): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
    }
    else if ( mode == 2 )
    {
    	clock_t t = clock();
    	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
    	normal_estimator.setNumberOfThreads( 12 );
    	normal_estimator.setSearchMethod ( tree );
    	normal_estimator.setInputCloud ( cloud_p );
    	normal_estimator.setRadiusSearch ( radius );
    	// normal_estimator.setKSearch (50);
    	normal_estimator.compute ( *normals );
    	std::cout << "normal estimation time (multi-thresd): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
    }
    else if ( mode == 3 )
    {
      clock_t t = clock();
      pcl::search::KdTree<pcl::PointXYZI>::Ptr search_tree (new pcl::search::KdTree<pcl::PointXYZI>);
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp (new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud (*cloud, *cloud_tmp);
      pcl::copyPointCloud (*cloud_tmp, *cloud_with_normals);
      pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointXYZINormal> normal_estimator;
      normal_estimator.setNumberOfThreads( 12 );
      normal_estimator.setSearchMethod ( search_tree );
      normal_estimator.setInputCloud ( cloud_tmp );
      normal_estimator.setRadiusSearch ( radius );
      //normal_estimator.setKSearch (50);
      normal_estimator.compute ( *cloud_with_normals );
      std::cout << "normal estimation time (multi-thresd): " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
    }
}

void Detector::getColoredCloud()
{
	if (!clusters.empty ())
	{
		colored_cloud = (new PointCloud)->makeShared ();
		
		srand (static_cast<unsigned int> (time (0)));
		std::vector<unsigned char> colors;
		for (size_t i_segment = 0; i_segment < clusters.size (); i_segment++)
		{
			colors.push_back (static_cast<unsigned char> (rand () % 256));
			colors.push_back (static_cast<unsigned char> (rand () % 256));
			colors.push_back (static_cast<unsigned char> (rand () % 256));
		}
		
		colored_cloud->width = cloud->width;
		colored_cloud->height = cloud->height;
		colored_cloud->is_dense = cloud->is_dense;
		for (size_t i_point = 0; i_point < cloud->points.size (); i_point++)
		{
			pcl::PointXYZRGB point;
			point.x = *(cloud->points[i_point].data);
			point.y = *(cloud->points[i_point].data + 1);
			point.z = *(cloud->points[i_point].data + 2);
			point.r = 0;
			point.g = 0;
			point.b = 0;
			colored_cloud->points.push_back (point);
		}
	
		std::vector< pcl::PointIndices >::iterator i_segment;
		int next_color = 0;
		for (i_segment = clusters.begin (); i_segment != clusters.end (); i_segment++)
		{
			std::vector<int>::iterator i_point;
			for (i_point = i_segment->indices.begin (); i_point != i_segment->indices.end (); i_point++)
			{
				int index;
				index = *i_point;
				colored_cloud->points[index].r = colors[3 * next_color];
				colored_cloud->points[index].g = colors[3 * next_color + 1];
				colored_cloud->points[index].b = colors[3 * next_color + 2];
      }
			next_color++;
		}
   }
}

void Detector::getClusters(int method)
{
	clock_t t = clock();

	if ( method == 1 ) // use depth based region growth only
	{
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (4000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud_p);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);

    //std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);
  }

  else if ( method == 2 )
  {
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.015); // 1cm
    ec.setMinClusterSize (50);
    ec.setMaxClusterSize (1000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_p);
    ec.extract (clusters);
  }

  else if ( method == 3 )
  {
    // Set up a Conditional Euclidean Clustering class
    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec (true);
    cec.setInputCloud (cloud_with_normals);
    cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (0.02);
    cec.setMinClusterSize (50);
    cec.setMaxClusterSize (1000);
    cec.segment (clusters);
  }

    std::cout << "segmentation time: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
}

// convert the segmentation results to vector of BoundingBox
void Detector::clusters2Bboxes()
{	
  clock_t t = clock();

	for( int i = 0; i < clusters.size(); i++ )
	{
		BoundingBox bbox;
		int min_x = std::numeric_limits<int>::max();
		int min_y = std::numeric_limits<int>::max();
		int max_x = std::numeric_limits<int>::min();
		int max_y = std::numeric_limits<int>::min();
		int idx;
		cv::Point3f point2d, point3d;

		for( int j = 0; j < clusters[i].indices.size(); j++ )
		{
			idx = clusters[i].indices[j];
			point3d.x = cloud->points[idx].x;
			point3d.y = cloud->points[idx].y;
			point3d.z = cloud->points[idx].z;

			point2d = point3dTo2d( point3d, camera );

			/*if (min_x > point2d.x ) { min_x = getMax(point2d.x, 0); }
			if (max_x < point2d.x ) { max_x = getMin(point2d.x, cloud->width); }
			if (min_y > point2d.y ) { min_y = getMax(point2d.y, 0); }
			if (max_y < point2d.y ) { max_y = getMin(point2d.y, cloud->height); }*/

			if (min_x > point2d.x ) { min_x = point2d.x; }
			if (max_x < point2d.x ) { max_x = point2d.x; }
			if (min_y > point2d.y ) { min_y = point2d.y; }
			if (max_y < point2d.y ) { max_y = point2d.y; }
		}
		bbox.min_x = min_x;
		bbox.min_y = min_y;
		bbox.max_x = max_x;
		bbox.max_y = max_y;

		if ( max_y-min_y >= 15 && max_x-min_x >= 15 && max_y-min_y <=200 && max_x-min_x <=200 )
    {
      bboxes.push_back( bbox );
      std_msgs::Int8 msgi;
      msgi.data = int(i+1);
      instance_id.push_back( msgi );
    }
	}
  std::cout << "clusters2Bboxes time consuming: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
}


void Detector::refineDepth ()
{
  clock_t t = clock();

  cv::Mat object_mask(depth.size(), CV_8UC1, cv::Scalar(0.0));
  int boundary = 20;
  int MAX_X = depth.cols - 1;
  int MAX_Y = depth.rows - 1;

  for(int i = 0; i < bboxes.size(); i++)
  {
    int edge_x = int(0.2*(bboxes[i].max_x-bboxes[i].min_x));
    int edge_y = int(0.2*(bboxes[i].max_y-bboxes[i].min_y));

    int shift_x = edge_x / 2;
    int shift_y = edge_y / 2;

    int min_x = std::max(bboxes[i].min_x - boundary + shift_x, 0);
    int max_x = std::min(bboxes[i].max_x + boundary + shift_x, MAX_X);
    int min_y = std::max(bboxes[i].min_y - boundary + shift_y, 0);
    int max_y = std::min(bboxes[i].max_y + boundary + shift_y, MAX_Y);

    object_mask(cv::Range(min_y,max_y), cv::Range(min_x,max_x)) = cv::Scalar(255.0);
    // cv::Mat bboxRoi = object_mask(cv::Rect(min_x, min_y, tmp_w, tmp_h));
    // set object regions (proposals) to 255.0
    // bboxRoi.setTo(cv::Scalar(255.0));
  }

  bool is_visualize = false;

  if( is_visualize )
  {
    char window_name[] = "object_mask";
    cv::namedWindow( window_name, CV_WINDOW_NORMAL );
    cv::resizeWindow( window_name, 400, 300 );
    //img.copyTo( dst, edge );
    cv::imshow( window_name, object_mask );

    // Wait until user exit program by pressing a key
    cv::waitKey(500);
  }

  odrFillNaNValues(rgb, depth, refined_depth, object_mask, 0.5);
  std::cout << "depth map refinement (Bilateral Filtering) time: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
}

// get segmentation map to show the segmented instances
void Detector::getInstanceMap ()
{
  clock_t t = clock(); 

  double resize_factor = 0.5;

  cv::Mat instance_map_, refined_depth_;

  cv::resize(instance_map, instance_map_, cv::Size(0,0), resize_factor, resize_factor);
  cv::resize(refined_depth, refined_depth_,  cv::Size(0,0), resize_factor, resize_factor);

  if (!clusters.empty())
  {
    std::vector<pcl::PointIndices>::iterator i_segment;

    int instance_id = 1; // start with 1. And 0 is the background

    cv::Point3f point3d, point2d;
    std::vector<int>::iterator i_point;
    int index;

    for (i_segment = clusters.begin(); i_segment != clusters.end(); i_segment++)
    {
      for (i_point = i_segment->indices.begin(); i_point != i_segment->indices.end(); i_point++)
      {
        index = *i_point;
        point3d.x = *(cloud->points[index].data);
        point3d.y = *(cloud->points[index].data + 1);
        point3d.z = *(cloud->points[index].data + 2);

        point2d = point3dTo2d( point3d, camera );
        point2d.x = round(point2d.x * resize_factor);
        point2d.y = round(point2d.y * resize_factor);

        int boundary = 3;
        //std::cout << point2d.x << " " << point2d.y << std::endl;
        if (point2d.x > boundary && point2d.x < instance_map.cols-boundary && point2d.y > boundary && point2d.y < instance_map.rows-boundary)
        {
          instance_map_.at<uchar>(int(point2d.y), int(point2d.x)) = uchar(instance_id);
        }

      }
      instance_id ++;
    } 
  }

  int iterNum = 3;
  cv::Mat edge = cv::Mat(refined_depth_.rows, refined_depth_.cols, CV_8UC1, cv::Scalar(0.));
  //char mode[] = "depth";
  //cannyEdgeDetection( depth, edge, mode );
  
  depthEdgeDetection( refined_depth_, edge, 0.02 ); // use refined_depth instead of original depth

  for( int i = 0; i < iterNum; i++ )
    fillHoles( instance_map_, edge );

  if (resize_factor != 1)
    resize(instance_map_, instance_map, original_size, 0, 0, CV_INTER_NN); //INTER_NEAREST

  if (0)
  {
      /// Create a window
      char window_name[] = "Instance Map";
      cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );
      //img.copyTo( dst, edge );
      cv::imshow( window_name, instance_map );

      /// Wait until user exit program by pressing a key
      cv::waitKey(0);
  }

  std::cout << "instance_map is obtained, time consuming: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;
}


// extract RGB features from object proposals
