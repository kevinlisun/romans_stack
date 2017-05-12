/*************************************************************************
	> File Name: src/utility.cpp
	> Author: Kevin Li Sun
	> Mail:  lisunsir@gmail.com
    > Implementation of utility.h
	> Created Time: 2016/07/08
 ************************************************************************/
# pragma once

// headfiles 
#include <fstream>
#include <vector>
#include <map>
using namespace std;

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include "highgui.h"

// ArrayFire
//#include <arrayfire.h>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <omp.h>

#include <time.h>

// point cloud type definition
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// camera intrisic paramers
struct CAMERA_INTRINSIC_PARAMETERS 
{ 
    double cx, cy, fx, fy, scale;
};


// fuction interface
// image2PonitCloud output PointCloud xyzrgb
void image2PointCloud( PointCloud::Ptr & cloud, cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );
// output PointCloud<pcl::PointXYZ>
void image2PointCloud( pcl::PointCloud <pcl::PointXYZ>::Ptr & cloud, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );
// output PointCloud<pcl::PointXYZ> with 2D coordinates list positions
void image2PointCloud( pcl::PointCloud <pcl::PointXYZ>::Ptr & cloud, std::vector<cv::Point2i> & positions, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );
// input surface normals, output dense cv::Mat in each <x,y> position stores 3 dimentional normal values
void normals2NormalMap( cv::Mat & normal_map, const pcl::PointCloud <pcl::Normal>::Ptr normals, std::vector<cv::Point2i> positions, int height, int width );

// convert 3D image point to 3D object point
// input: 2D point <u,v> and depth <d>, Point3f (u,v,d)
cv::Point3f point2dTo3d( cv::Point3f point, CAMERA_INTRINSIC_PARAMETERS camera );

// convert 3D object points to 2D image points (with depth)
// input: 3D points (x,y,z)
cv::Point3f point3dTo2d( cv::Point3f point, CAMERA_INTRINSIC_PARAMETERS camera );

// get the median value of a std::vector
//int getMedian(std::vector<int> array);
template<typename T>
T getMedian( std::vector<T> array );

double get3Ddistance( cv::Point3f p0, cv::Point3f p1, const CAMERA_INTRINSIC_PARAMETERS camera );

// fill the holes in a sparse semantic map
void fillHoles( cv::Mat & img ); 
void fillHoles( cv::Mat & img, const cv::Mat edge ); // with edge protection
// erode the semantic map to thin the boundary
void imgErosion( cv::Mat & img, int erosion_type );
// Canny edge detection
void cannyEdgeDetection( cv::Mat & rgb, cv::Mat & mask, char* mode );
// edge detection for depth map
void depthEdgeDetection( cv::Mat & depth, cv::Mat & edge, double threshold );
// fill the nan (0) values in depth map using median filtering
void refineDepth( const cv::Mat img, cv::Mat &img_new, int radius );

// GPU
/*af::array medianFilter( const af::array &in, int window_width, int window_height );
void medianFilter( cv::Mat & input, int window_width, int window_height, int type );
void mat2array( const cv::Mat& input, af::array & output );
void array2mat( const af::array & input, cv::Mat & output, int type );*/


// Camera parameter reader
class ParameterReader
{
public:
    ParameterReader( string filename="/home/kevin/catkin_ws/src/romans_stack/odr/param/parameters.txt" )
    {
        ifstream fin( filename.c_str() );
        if (!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return;
        }
        while(!fin.eof())
        {
            string str;
            getline( fin, str );
            if (str[0] == '#')
            {
                // if ‘＃’, it is a line of comment
                continue;
            }

            int pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr( 0, pos );
            string value = str.substr( pos+1, str.length() );
            data[key] = value;

            if ( !fin.good() )
                break;
        }
    }
    string getData( string key )
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<endl;
            return string("NOT_FOUND");
        }
        return iter->second;
    }
public:
    map<string, string> data;
};

inline static CAMERA_INTRINSIC_PARAMETERS getCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = atof( pd.getData( "camera.fx" ).c_str());
    camera.fy = atof( pd.getData( "camera.fy" ).c_str());
    camera.cx = atof( pd.getData( "camera.cx" ).c_str());
    camera.cy = atof( pd.getData( "camera.cy" ).c_str());
    camera.scale = atof( pd.getData( "camera.scale" ).c_str() );
    return camera;
}

inline static CAMERA_INTRINSIC_PARAMETERS getDefaultCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera; // default intrisic parameters for kinect v2
    camera.cx=473.5111188081456;
    camera.cy=267.62102832727044;
    camera.fx=527.8462664548338;
    camera.fy=527.7365063721007;
    camera.scale = 1000.0;
    return camera;
}

inline static CAMERA_INTRINSIC_PARAMETERS getVirtualCamera()
{
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera; // default intrisic parameters for kinect v2
    camera.cx = 2.5e+02;
    camera.cy = 2.5e+02;
    camera.fx = 5.0e+02;
    camera.fy = 5.0e+02;
    camera.scale = 1.0;
    return camera;
}






