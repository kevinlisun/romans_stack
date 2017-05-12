/*************************************************************************
	> File Name: pcl_plus.h
	> Author: Kevin Li Sun
	> Mail: lisunsir@gmail.com
	> Created Time: 2016/09/15
 ************************************************************************/

#include "pcl_plus.h"
#include "utility.h"

void ppDownSampling ( PointCloud::Ptr & cloud, double leaf_size )
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

void ppFiltering ( PointCloud::Ptr & cloud )
{
    // remove NaN values
    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::removeNaNFromPointCloud(*cloud, *cloud, *indices);

    /*pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_p);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, max_dist);
    pass.filter (*indices);*/
}

int user_data;
    
void viewerOneOff (pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor (1.0, 0.5, 1.0);
    pcl::PointXYZ o;
    o.x = 1.0;
    o.y = 0;
    o.z = 0;
    viewer.addSphere (o, 0.25, "sphere", 0);
    std::cout << "i only run once" << std::endl;
    
}
    
void viewerPsycho (pcl::visualization::PCLVisualizer& viewer)
{
    static unsigned count = 0;
    std::stringstream ss;
    ss << "Once per viewer loop: " << count++;
    viewer.removeShape ("text", 0);
    viewer.addText (ss.str(), 200, 300, "text", 0);
    
    //FIXME: possible race condition here:
    user_data++;
}

void ppViewer(PointCloud::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    
    //blocks until the cloud is actually rendered
    viewer.showCloud(cloud);

    //use the following functions to get access to the underlying more advanced/powerful
    //PCLVisualizer
    
    //This will only get called once
    viewer.runOnVisualizationThreadOnce (viewerOneOff);
    
    //This will get called once per visualization iteration
    //viewer.runOnVisualizationThread (viewerPsycho);
    while (!viewer.wasStopped ())
    {
    //you can also do cool processing here
    //FIXME: Note that this is running in a separate thread from viewerPsycho
    //and you should guard against race conditions yourself...
    user_data++;
    }
}

// get segmentation map to show the segmented instances
void ppGetInstanceMap (PointCloud::Ptr cloud, std::vector<pcl::PointIndices> clusters, cv::Mat& seg_map)
{
  if (!clusters.empty())
  {
    std::vector<pcl::PointIndices>::iterator i_segment;
    int instance_id = 1; // start with 1. And 0 is the background

    for (i_segment = clusters.begin(); i_segment != clusters.end(); i_segment++)
    {
      std::vector<int>::iterator i_point;
      for (i_point = i_segment.indices.begin(); i_point != i_segment.indices.end(); i_point++)
      {
        cv::Point3f point;
        point.x = *(cloud->points[i_point].data);
        point.y = *(cloud->points[i_point].data + 1);
        point.z = *(cloud->points[i_point].data + 2);

        point = point3dTo2d( point, camera );
        seg_map.at<cv::Vec3f>(point.y, point.x).val[0] = instance_id;
      }
    }
  }
}

// get a colored cloud to show the segmentation result
void ppGetColoredCloud (PointCloud::Ptr cloud, std::vector<pcl::PointIndices> clusters, PointCloud::Ptr & colored_cloud)
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


void ppNormalEstimation( pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_p, pcl::PointCloud <pcl::Normal>::Ptr & normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree, double radius )
{

    clock_t t = clock();
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod ( tree );
    normal_estimator.setInputCloud ( cloud_p );
    normal_estimator.setRadiusSearch ( radius );
    // normal_estimator.setKSearch (50);
    normal_estimator.compute ( *normals );

    std::cout << "normal time: " << clock() - t << std::endl;
    t = clock();
}

void ppRegSegmentation( PointCloud::Ptr cloud_p, std::vector <pcl::PointIndices> & clusters, pcl::PointCloud <pcl::Normal>::Ptr & normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr & tree, bool is_visualization )
{

    clock_t t = clock();

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize (50);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud_p);
    //reg.setIndices (indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (5.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (0.5);

    //std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    std::cout << "rg time: " << clock() - t << std::endl;

    if (is_visualization)
    {
    	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
    	pcl::visualization::CloudViewer viewer ("Cluster viewer");
    	viewer.showCloud(colored_cloud);
    	while (!viewer.wasStopped ())
    	{
    	}
    }

}


void ppDonSegmentation( PointCloud::Ptr cloud, std::vector <pcl::PointIndices> & clusters, bool is_visualization )
{
	double scale1 = 0.03;
	double scale2 = 0.05;
	double threshold = 0.2;
	double segradius = 0.1;

	if (scale1 >= scale2)
  	{
    	cerr << "Error: Large scale must be > small scale!" << endl;
    	exit (EXIT_FAILURE);
  	}

	  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p (new pcl::PointCloud<pcl::PointXYZ>);

    copyPointCloud(*cloud, *cloud_p);

    pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud_p);
    //normal_estimator.setKSearch (50);

  	// Compute normals using both small and large scales at each point
  	/**
   	* NOTE: setting viewpoint is very important, so that we can ensure
   	* normals are all pointed in the same direction!
   	*/
  	normal_estimator.setViewPoint (0, 0, 0);

  	// calculate normals with the small scale
  	cout << "Calculating normals for scale..." << scale1 << endl;
    pcl::PointCloud <pcl::PointNormal>::Ptr normals_small_scale (new pcl::PointCloud <pcl::PointNormal>);
  	normal_estimator.setRadiusSearch (scale1);
 	normal_estimator.compute (*normals_small_scale);

  	// calculate normals with the large scale
  	cout << "Calculating normals for scale..." << scale2 << endl;
    pcl::PointCloud <pcl::PointNormal>::Ptr normals_large_scale (new pcl::PointCloud <pcl::PointNormal>);

  	normal_estimator.setRadiusSearch (scale2);
  	normal_estimator.compute (*normals_large_scale);

  	// Create output cloud for DoN results
  	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud (new pcl::PointCloud<pcl::PointNormal>);
  	copyPointCloud(*cloud_p, *doncloud);

  	cout << "Calculating DoN... " << endl;
  	// Create DoN operator
  	pcl::DifferenceOfNormalsEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::PointNormal> don;
  	don.setInputCloud (cloud_p);
  	don.setNormalScaleLarge (normals_large_scale);
  	don.setNormalScaleSmall (normals_small_scale);

  	if (!don.initCompute ())
  	{
    	std::cerr << "Error: Could not intialize DoN feature operator" << std::endl;
    	exit (EXIT_FAILURE);
  	}

  	// Compute DoN
  	don.computeFeature (*doncloud);

  	// Filter by magnitude
  	cout << "Filtering out DoN mag <= " << threshold << "..." << endl;

  	// Build the condition for filtering
  	pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond (
    	new pcl::ConditionOr<pcl::PointNormal> ()
    );
  	range_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (
        new pcl::FieldComparison<pcl::PointNormal> ("curvature", pcl::ComparisonOps::GT, threshold))
    );
  	// Build the filter
  	pcl::ConditionalRemoval<pcl::PointNormal> condrem (range_cond);
  	condrem.setInputCloud (doncloud);

  	pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered (new pcl::PointCloud<pcl::PointNormal>);

  	// Apply filter
  	condrem.filter (*doncloud_filtered);

  	doncloud = doncloud_filtered;

  	// Save filtered output
  	std::cout << "Filtered Pointcloud: " << doncloud->points.size () << " data points." << std::endl;

  	// Filter by magnitude
  	cout << "Clustering using EuclideanClusterExtraction with tolerance <= " << segradius << "..." << endl;

  	pcl::search::KdTree<pcl::PointNormal>::Ptr segtree (new pcl::search::KdTree<pcl::PointNormal>);
  	segtree->setInputCloud (doncloud);

  	//std::vector<pcl::PointIndices> clusters;
  	pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;

  	ec.setClusterTolerance (segradius);
  	ec.setMinClusterSize (50);
  	ec.setMaxClusterSize (100000);
  	ec.setSearchMethod (segtree);
  	ec.setInputCloud (doncloud);
  	ec.extract (clusters);

}

void ppColorRegSegmentation(PointCloud::Ptr cloud, std::vector <pcl::PointIndices> & clusters, bool is_visualization )
{


  	pcl::IndicesPtr indices (new std::vector <int>);
  	pcl::PassThrough<pcl::PointXYZRGB> pass;
  	pass.setInputCloud (cloud_p);
  	pass.setFilterFieldName ("z");
  	pass.setFilterLimits (0.0, 1.5);
  	pass.filter (*indices);

  	pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  	reg.setInputCloud (cloud_p);
  	reg.setIndices (indices);
    //std::cout << indices << std::endl;
  	reg.setSearchMethod (tree);
  	reg.setDistanceThreshold (10);
  	reg.setPointColorThreshold (6);
  	reg.setRegionColorThreshold (5);
  	reg.setMinClusterSize (100);

  	//std::vector <pcl::PointIndices> clusters;
  	reg.extract (clusters);

  	if ( is_visualization )
  	{
  		pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  		pcl::visualization::CloudViewer viewer ("Cluster viewer");
  		viewer.showCloud (colored_cloud);
  		while (!viewer.wasStopped ())
  		{
    		boost::this_thread::sleep (boost::posix_time::microseconds (100));
		}
	}
}


