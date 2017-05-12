#include <iostream>
#include "utility.h"
#include "pcl_plus.h"
#include "detector.h"

#include <string>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>


int main(int argc, char** argv)
{
    // read./data/rgb.png and ./data/depth.png，and convert to point cloud

    // define rgb and depth
    cv::Mat rgb, depth;
    // read image using cv::imread()
    
    rgb = cv::imread( "/home/kevin/romans_ws/src/odr/data/rgb.tiff" );
    // rgb is 8UC3的
    // depth is 16UC1
    depth = cv::imread( "/home/kevin/romans_ws/src/odr/data/depth.tiff", -1 );

    Detector detector( rgb, depth, 0.5 );
    detector.estimateNormal(1, 0.03);
    detector.getClusters(1);
    detector.clusters2Bboxes();
    detector.getColoredCloud();

    std::cout << *(detector.colored_cloud) << std::endl;

    std::cout << "Number of object proposals: " << detector.clusters.size() << std::endl;

    //CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    //PointCloud::Ptr cloud = image2PointCloud(rgb, depth, camera);

    odrViewer(detector.colored_cloud);
    //std::vector <pcl::PointIndices> clusters;
    //ppColorRegSegmentation(cloud, clusters);

    //std::vector<BoundingBox> bbox;
    //odrClusters2Bboxes(cloud, clusters, bbox);

    return 0;
}
