
/*
* @Author: Kevin Sun
* @Date:   2016-11-18 14:36:53
* @Last Modified by:   Kevin Sun
* @Last Modified time: 2016-11-18 14:58:58
*/

#include "utility.h"

cv::Vec3i getColor(uchar label);
cv::Mat getColoredSemanticMap(cv::Mat & smap);


cv::Vec3i getColor(int label)
{
	cv::Vec3i color;

	switch (label)
	{
		case 0 : { color.val[0] = 100; color.val[1] = 100; color.val[2] = 100; } break;  // nackground
		case 1 : { color.val[0] = 255; color.val[1] = 0; color.val[2] = 0; } break;     // blue bottle
		case 2 : { color.val[0] = 0; color.val[1] = 255; color.val[2] = 0; } break;     // green cans
		case 3 : { color.val[0] = 0; color.val[1] = 0; color.val[2] = 255; } break;     //red  chains
		case 4 : { color.val[0] = 0; color.val[1] = 255; color.val[2] = 255; } break;   // yellow cloth
		case 5 : { color.val[0] = 128; color.val[1] = 0; color.val[2] = 128; } break;   // purple gloves
		case 6 : { color.val[0] = 0; color.val[1] = 0; color.val[2] = 0; } break; // black  metal objects
		case 7 : { color.val[0] = 255; color.val[1] = 255; color.val[2] = 255; } break;   // white pipejoints
		case 8 : { color.val[0] = 0; color.val[1] = 64; color.val[2] = 0; } break;   // dark green plastic pipes
		case 9 : { color.val[0] = 192; color.val[1] = 192; color.val[2] = 0; } break; // light blue sponges
		case 10 : { color.val[0] = 0; color.val[1] = 128; color.val[2] = 255; } break;  // orange  wood block 

	}

	return color;
}

cv::Mat getColoredSemanticMap(cv::Mat & s_map)
{
	int cols = s_map.cols;
	int rows = s_map.rows;

	cv::Mat c_map(rows, cols, CV_8UC3, cv::Scalar(0.));

	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < cols; j++ )
		{
			cv::Vec3i color = getColor(int(s_map.at<uchar>(i,j)));
			c_map.ptr<uchar>(i)[j*3] = uchar(color.val[0]);
			c_map.ptr<uchar>(i)[j*3+1] = uchar(color.val[1]);
			c_map.ptr<uchar>(i)[j*3+2] = uchar(color.val[2]);
		}
	}		
	//c_map.at<cv::Vec3i>(i,j) = getColor( s_map.at<uchar>(i,j) );

	return c_map;
}