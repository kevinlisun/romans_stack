/*************************************************************************
	> File Name: src/utility.cpp
	> Author: Kevin Li Sun
	> Mail:  lisunsir@gmail.com
    > Implementation of utility.h
	> Created Time: 2016/07/08
 ************************************************************************/

#include "utility.h"


void image2PointCloud( pcl::PointCloud <pcl::PointXYZ>::Ptr & cloud, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // get depth value at (m,n)
            float d = depth.ptr<float>(m)[n];
            // if d has no value
            if (d == 0)
                continue;
            // if d has value, add a new point
            pcl::PointXYZ p;

            // project 2D to 3D
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            //std::cout << p.x << " " << p.y << " " << p.z << std::endl;
            // push the point p to the cloud
            cloud->points.push_back( p );
        }
    // cloud setting up
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
}

void image2PointCloud( pcl::PointCloud <pcl::PointXYZ>::Ptr & cloud, std::vector<cv::Point2i> & positions, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    positions.resize(0);

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // get depth value at (m,n)
            float d = depth.ptr<float>(m)[n];
            // if d has no value
            if (d == 0)
                continue;
            // if d has value, add a new point
            pcl::PointXYZ p;

            // project 2D to 3D
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            // std::cout << p.x << " " << p.y << " " << p.z << std::endl;
            // push the point p to the cloud
            cloud->points.push_back( p );

            // record <x,y> coordinates
            cv::Point2i pos;

            pos.x = n;
            pos.y = m;

            positions.push_back( pos );
            // std::cout << pos << std::endl;
        }
    // cloud setting up

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
}

void image2PointCloud( PointCloud::Ptr & cloud, cv::Mat & rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // get depth value in <m,n>
            float d = depth.ptr<float>(m)[n];
            // if d is nan value (0)
            //if (d == 0)
            //    continue;
            // d is not 0
            PointT p;

            // caculate the 3D coordinates p<x,y,z> from 2D <m,n,d>
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            //std::cout << p.x << " " << p.y << " " << p.z << std::endl;
            
            // get RGB color values
            // rgb is in 3 channel BGR format，use b g r order here
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // add p to the cloud
            cloud->points.push_back( p );
        }
    // seting of the cloud
    cloud->height = depth.rows;
    cloud->width = depth.cols;
    cloud->is_dense = true;
}

void normals2NormalMap( cv::Mat & normal_map, const pcl::PointCloud <pcl::Normal>::Ptr normals, std::vector<cv::Point2i> positions, int height, int width )
{
    if ( normals->points.size() != positions.size() )
    {
        std::cout << "WARNING: Size of the normals and position are not equal!" << std::endl;
        std::cout << normals->points.size() << " vs " <<  positions.size() << std::endl;
    }
    else
    {
        std::cout << "Size checking is passthrough." << std::endl;
        std::cout << normals->points.size() << " vs " <<  positions.size() << std::endl;
    }

    // cv::Mat normal_map(height, width, CV_32FC3, cv::Scalar(0.));

    for (int i = 0; i < positions.size(); i++ )
    {
        normal_map.at<cv::Vec3f>(positions[i].y, positions[i].x).val[0] = float(normals->points[i].normal_x);
        normal_map.at<cv::Vec3f>(positions[i].y, positions[i].x).val[1] = float(normals->points[i].normal_y);
        normal_map.at<cv::Vec3f>(positions[i].y, positions[i].x).val[2] = float(normals->points[i].normal_z);
        // std::cout << float(normals->points[i].normal_x) << " " << float(normals->points[i].normal_y) << " " << float(normals->points[i].normal_z) << std::endl;
    }

    // std::cout << normal_map << std::endl;
}

cv::Point3f point2dTo3d( cv::Point3f point, CAMERA_INTRINSIC_PARAMETERS camera )
{
    cv::Point3f p; // 3D point
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

cv::Point3f point3dTo2d( cv::Point3f point, CAMERA_INTRINSIC_PARAMETERS camera )
{
    cv::Point3f p; // 2D point p(x,y,depth)
    p.z = point.z * camera.scale;
    if(point.z != 0)
    {
        p.x = (point.x * camera.fx) / point.z + camera.cx;
        p.y = (point.y * camera.fy) / point.z + camera.cy;
    }    
    return p;
}

template<typename T>
T getMedian( std::vector<T> array )
{
    T median;
    size_t size = array.size();

    std::sort(array.begin(), array.end());

    if (size  % 2 == 0)
    {
        median = (array[size / 2 - 1] + array[size / 2]) / 2;
    }
    else 
    {
        median = array[size / 2];
    }

    return median;
}

double get3Ddistance( cv::Point3f p0, cv::Point3f p1, const CAMERA_INTRINSIC_PARAMETERS camera )
{
    cv::Point3f p0_3d = point2dTo3d( p0, camera );
    cv::Point3f p1_3d = point2dTo3d( p1, camera );

    double dist = norm( cv::Mat(p0_3d), cv::Mat(p1_3d), cv::NORM_L2 );

    return dist;
}

void fillHoles( cv::Mat & img )
{
    cv::Mat img_new (img.rows, img.cols, CV_8UC1, cv::Scalar(0.));
    std::vector<int> vec;

    for ( int i = 1; i < img.rows-1; i++ )
    {
        for ( int j = 1; j < img.cols-1; j++ )
        {
            if (int(img.at<uchar>(i,j)) != 0)
            {
                img_new.at<uchar>(i,j) = img.at<uchar>(i,j);
                continue;
            }

            int array[9];
            double distances[9];
            
            array[0] = int(img.at<uchar>(std::max(i-1,0),j)); //up
            array[1] = int(img.at<uchar>(std::min(i+1,img.rows),j)); //down
            array[2] = int(img.at<uchar>(i,std::max(j-1,0))); //left
            array[3] = int(img.at<uchar>(i,std::min(j+1,img.cols))); // right
            array[4] = int(img.at<uchar>(std::max(i-1,0),std::max(j-1,0))); //upper left
            array[5] = int(img.at<uchar>(std::max(i-1,0),std::min(j+1,img.cols))); // upper right
            array[6] = int(img.at<uchar>(std::min(i+1,img.rows),std::max(j-1,0))); //lower left
            array[7] = int(img.at<uchar>(std::min(i+1,img.rows),std::min(j+1,img.cols))); //lower right
            array[8] = int(img.at<uchar>(i,j));

            vec.resize(0);

            for ( int k = 0; k < 9; k++ )
            {         
                if ( array[k] > 0 )
                    vec.push_back( array[k] );
            }

            if (vec.size() == 0)
                img_new.at<uchar>(i,j) = uchar(0);
            else
                img_new.at<uchar>(i,j) = uchar(getMedian(vec));
        }
    }

    for ( int i = 0; i < img.rows; i++ )
        for ( int j = 0; j < img.cols; j++ )
            img.at<uchar>(i,j) = img_new.at<uchar>(i,j);
}

void fillHoles( cv::Mat & img, const cv::Mat edge )
{
    cv::Mat img_new (img.rows, img.cols, CV_8UC1, cv::Scalar(0.));
    std::vector<int> vec;

    /*int this_thread = omp_get_thread_num(), num_threads = omp_get_num_threads();
    int my_start = (this_thread  ) * 10 / num_threads;
    int my_end   = (this_thread+1) * 10 / num_threads;
    for(int n=my_start; n<my_end; ++n)
        printf(" %d", n);*/

    int i, threadID;

    #pragma omp parallel for private(i, threadID)
    for ( i = 1; i < img.rows-1; i++ )
    {
        threadID = omp_get_thread_num();
        #pragma omp critical
        {
        //printf("Thread %d reporting\n", threadID);

        for ( int j = 1; j < img.cols-1; j++ )
        {
            if ( int(img.at<uchar>(i,j)) != 0 || int(edge.at<uchar>(i,j)) != 0 )
            {
                img_new.at<uchar>(i,j) = img.at<uchar>(i,j);
                continue;
            }

            int array[9];
            double distances[9];
            
            array[0] = int(img.at<uchar>(std::max(i-1,0),j)); //up
            array[1] = int(img.at<uchar>(std::min(i+1,img.rows),j)); //down
            array[2] = int(img.at<uchar>(i,std::max(j-1,0))); //left
            array[3] = int(img.at<uchar>(i,std::min(j+1,img.cols))); // right
            array[4] = int(img.at<uchar>(std::max(i-1,0),std::max(j-1,0))); //upper left
            array[5] = int(img.at<uchar>(std::max(i-1,0),std::min(j+1,img.cols))); // upper right
            array[6] = int(img.at<uchar>(std::min(i+1,img.rows),std::max(j-1,0))); //lower left
            array[7] = int(img.at<uchar>(std::min(i+1,img.rows),std::min(j+1,img.cols))); //lower right
            array[8] = int(img.at<uchar>(i,j));

            vec.resize(0);

            for ( int k = 0; k < 9; k++ )
            {         
                if ( array[k] > 0 )
                    vec.push_back( array[k] );
            }

            if (vec.size() == 0)
                img_new.at<uchar>(i,j) = uchar(0);
            else
                img_new.at<uchar>(i,j) = uchar(getMedian(vec));
        }
        }
    }

    //for ( int i = 0; i < img.rows; i++ )
    //    for ( int j = 0; j < img.cols; j++ )
    //        img.at<uchar>(i,j) = img_new.at<uchar>(i,j);
    img_new.copyTo(img);
}

void imgErosion( cv::Mat & img, int erosion_elem )
{ 
    int erosion_type;
    int erosion_size = 0;

    if( erosion_elem == 0 ){ erosion_type = cv::MORPH_RECT; }
    else if( erosion_elem == 1 ){ erosion_type = cv::MORPH_CROSS; }
    else if( erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }

    cv::Mat element = cv::getStructuringElement( erosion_type,
                                       cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       cv::Point( erosion_size, erosion_size ) );
    /// Apply the erosion operation
    cv::erode( img, img, element ); // src dest
}

void cannyEdgeDetection( cv::Mat & img, cv::Mat & edge, char* mode )
{

  int edgeThresh = 1;
  int lowThreshold;
  int const max_lowThreshold = 10;
  int ratio = 3;
  int kernel_size = 3;
  cv::Mat gray;
  bool flag = 1;

  if ( strcmp(mode, "rgb") == 0 )
  {
    //Convert the image to grayscale
    cv::cvtColor( img, img, CV_BGR2GRAY );
    
  }
  else if ( strcmp(mode, "depth") == 0 )
  {
    cv::normalize( img, img, 500, 800, 4, CV_8UC1 );
  }
  else
    std::cout << "ERROR: Unknown data type.";
  
  cv::blur( img, edge, cv::Size(3,3) );

  std::cout << edge.depth() << std::endl;

  /// Canny detector
  Canny( edge, edge, lowThreshold, lowThreshold*ratio, kernel_size );


  if (flag)
  {
    /// Using Canny's output as a mask, we display our result
    cv::Mat dst;

    /// Create a window
    char window_name[] = "Edge Map";
    cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );
    //img.copyTo( dst, edge );
    cv::imshow( window_name, edge );

    /// Wait until user exit program by pressing a key
    cv::waitKey(0);
  }
}

void depthEdgeDetection( cv::Mat & img, cv::Mat & edge, double threshold )
{
    bool flag = 0;

    //cv::blur( img, img, cv::Size(5,5) );

    for ( int i = 1; i < img.rows-1; i++ )
    {
        for ( int j = 1; j < img.cols-1; j++ )
        {
            int array[4];
            array[0] = img.at<float>(std::max(i-1,0),j); //up
            array[1] = img.at<float>(std::min(i+1,img.rows),j); //down
            array[2] = img.at<float>(i,std::max(j-1,0)); //left
            array[3] = img.at<float>(i,std::min(j+1,img.cols)); // right

            for( int k = 0; k < 4; k++ )
            {
                if( std::abs(array[k]-img.at<float>(i,j)) > threshold*1000 )
                {
                    //std::cout << std::abs(array[k]-img.at<float>(i,j)) << " ";
                    edge.at<uchar>(i,j) = 255;
                    break;
                }
            }
        }
    }

    if (flag)
    {
        /// Create a window
        char window_name[] = "Edge Map";
        cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );
        //img.copyTo( dst, edge );
        cv::imshow( window_name, edge );

        /// Wait until user exit program by pressing a key
        cv::waitKey(0);
    }
}

void refineDepth( const cv::Mat img, cv::Mat &img_new, int radius )
{
    img_new = img.clone();
    std::vector<float> vec;

    bool flag = 0;

    for ( int i = 1; i < img.rows-1; i++ )
    {
        for ( int j = 1; j < img.cols-1; j++ )
        {
            if (img_new.at<float>(i,j) != 0)
                continue;

            int left = std::max(j-radius,0);
            int top = std::max(i-radius,0);
            int p_width = std::min(2*radius+1, img.cols-left);
            int p_height = std::min(2*radius+1, img.rows-top);

            //std::cout << left << " " << top << "  " <<  p_width << " " << p_height << std::endl;

            cv::Mat patch = img(cv::Rect(left, top, p_width, p_height));

            vec.resize(0);

            for ( int m = 0; m < patch.rows; m++ )
            {
                for ( int n = 0; n < patch.cols; n++ )
                {
                    if(patch.at<float>(m,n) != 0)
                        vec.push_back(patch.at<float>(m,n));
                }
            }
            if (vec.size() > 0)
                img_new.at<float>(i,j) = getMedian(vec);
        }
    }

    //img_new.copyTo(img);

    if (flag)
    {
        /// Create a window
        char window_name[] = "Depth Map";
        cv::namedWindow( window_name, CV_WINDOW_AUTOSIZE );
        //img.copyTo( dst, edge );

        double min;
        double max;
        cv::minMaxIdx(img, &min, &max);
        cv::Mat adjMap;
        cv::convertScaleAbs(img, adjMap, 255 / max);
        cv::imshow( window_name, adjMap );

        /// Wait until user exit program by pressing a key
        cv::waitKey(0);
    }

}


// GPU part pls comment these if you don't have a GPU device
/*
af::array medianFilter(const af::array & input, int window_width, int window_height)
{
    af::array output(input.dims());
    output(af::span, af::span, 0) = af::medfilt(input(af::span, af::span, 0), window_width, window_height);

    return output;
}

void medianFilter(cv::Mat & cv_mat, int window_width, int window_height, int type)
{
    std::cout << "here is okay 0" << std::endl;
    af::array array_mat;
    std::cout << "here is okay 1" << std::endl;
    mat2array(cv_mat, array_mat ); // convert to arrayfire 
    std::cout << "here is okay 2" << std::endl;
    array_mat(af::span, af::span, 0) = af::medfilt(array_mat(af::span, af::span, 0), window_width, window_height);
    std::cout << "here is okay 3" << std::endl;
    cv::Mat output = cv_mat.clone();
    array2mat(array_mat, cv_mat, type); //convert arrayfire to cv::Mat
    std::cout << "here is okay 4" << std::endl;
}

// convert cv::Mat to gpu af::array
void mat2array(const cv::Mat & input, af::array & output )
{
    std::cout << "here is okay 0/0" << std::endl;
    
    const unsigned size = input.rows * input.cols;
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    const unsigned channels = input.channels();

    std::cout << "here is okay 0/1" << std::endl;

    int tmp = 0;

    if (channels == 1)
    {
        std::cout << "here is okay 0/2" << std::endl;
        input.convertTo(input, CV_32F);
        std::cout << "here is okay 0/3" << std::endl;
        output = af::array(w, h, input.ptr<float>(0)).T();
    }
    else if (channels == 2)
    {
        input.convertTo(input, CV_32FC3);
        vector<cv::Mat> rgb;
        cv::split(input, rgb);
        output = af::array(w, h, 3);
        output(af::span, af::span, 0) = af::array(w, h, rgb[2].ptr<float>(0)).T();
        output(af::span, af::span, 1) = af::array(w, h, rgb[1].ptr<float>(0)).T();
        output(af::span, af::span, 2) = af::array(w, h, rgb[0].ptr<float>(0)).T();
    }
    else
    {
        std::cout << "ERROR: Unknown channels number." << std::endl;
    }
}

//convert gpu af::array to cpu cv::Mat
void array2mat(const af::array& input, cv::Mat& output, int type)
{
    output = cv::Mat(input.dims(0), input.dims(1), type);
    if (type == CV_32F)
    {
        float* data = output.ptr<float>(0);
        input.T().host((void*)data);
    }
    else if (type == CV_64F)
    {
        double* data = output.ptr<double>(0);
        input.T().as(f64).host((void*)data);        
    }
    else if (type == CV_8U)
    {
        uchar* data = output.ptr<uchar>(0);
        input.T().as(b8).host((void*)data);
    }
    else
    {
        std::cout << "ERROR: Unknown channels number." << std::endl;
    }
}
*/