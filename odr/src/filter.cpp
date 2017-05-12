/*******************************************************
* Copyright (c) 2015, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <arrayfire.h>

#include <iostream>
#include <time.h>

using namespace af;
array hurl(const array &in, int randomization, int repeat)
{
    int w = in.dims(0);
    int h = in.dims(1);
    float f = randomization / 100.0f;
    int dim = (int)(f*w*h);
    array ret_val = in.copy();
    array temp = moddims(ret_val, w*h, 3);
    for (int i = 0; i<repeat; ++i) {
        array idxs = (w*h)  * randu(dim);
        array rndR = 255.0f * randu(dim);
        array rndG = 255.0f * randu(dim);
        array rndB = 255.0f * randu(dim);
        temp(idxs, 0) = rndR;
        temp(idxs, 1) = rndG;
        temp(idxs, 2) = rndB;
    }
    ret_val = moddims(temp, in.dims());
    return ret_val;
}
array getRandomNeighbor(const array &in, int windW, int windH)
{
    array rnd = 2.0f*randu(in.dims(0), in.dims(1)) - 1.0f;
    array sx = seq(in.dims(0));
    array sy = seq(in.dims(1));
    array vx = tile(sx, 1, in.dims(1)) + floor(rnd*windW);
    array vy = tile(sy.T(), in.dims(0), 1) + floor(rnd*windH);
    array vxx = clamp(vx, 0, in.dims(0));
    array vyy = clamp(vy, 0, in.dims(1));
    array in2 = moddims(in, vx.elements(), 3);
    return moddims(in2(vyy*in.dims(0) + vxx, span), in.dims());
}
array spread(const array &in, int window_width, int window_height)
{
    return getRandomNeighbor(in, window_width, window_height);
}
array pick(const array &in, int randomization, int repeat)
{
    int w = in.dims(0);
    int h = in.dims(1);
    float f = randomization / 100.0f;
    int dim = (int)(f*w*h);
    array ret_val = in.copy();
    for (int i = 0; i<repeat; ++i) {
        array idxs = (w*h)  * randu(dim);
        array rnd = getRandomNeighbor(ret_val, 1, 1);
        array temp_src = moddims(rnd, w*h, 3);
        array temp_dst = moddims(ret_val, w*h, 3);
        temp_dst(idxs, span) = temp_src(idxs, span);
        ret_val = moddims(temp_dst, in.dims());
    }
    return ret_val;
}
void prewitt(array &mag, array &dir, const array &in)
{
    static float h1[] = { 1, 1, 1 };
    static float h2[] = { -1, 0, 1 };
    static array h1d(3, h1);
    static array h2d(3, h2);
    // Find the gradients
    array Gy = af::convolve(h2d, h1d, in) / 6;
    array Gx = af::convolve(h1d, h2d, in) / 6;
    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}
void sobelFilter(array &mag, array &dir, const array &in)
{
    // Find the gradients
    array Gy, Gx;
    af::sobel(Gx, Gy, in);
    // Find magnitude and direction
    mag = hypot(Gx, Gy);
    dir = atan2(Gy, Gx);
}
void normalizeImage(array &in, float min, float max)
{
    in = 255.0f*((in - min) / (max - min));
}
array DifferenceOfGaussian(const array &in, int window_radius1, int window_radius2)
{
    array ret_val;
    int w1 = 2 * window_radius1 + 1;
    int w2 = 2 * window_radius2 + 1;
    array g1 = gaussianKernel(w1, w1);
    array g2 = gaussianKernel(w2, w2);
    ret_val = (convolve(in, g1) - convolve(in, g2));
    return ret_val;
}
array medianfilter(const array &in, int window_width, int window_height)
{
    array ret_val(in.dims());
    ret_val(span, span, 0) = medfilt(in(span, span, 0), window_width, window_height);
    //ret_val(span, span, 1) = medfilt(in(span, span, 1), window_width, window_height);
    //ret_val(span, span, 2) = medfilt(in(span, span, 2), window_width, window_height);
    return ret_val;
}
array gaussianblur(const array &in, int window_width, int window_height, double sigma)
{
    array g = gaussianKernel(window_width, window_height, sigma, sigma);
    return convolve(in, g);
}
array emboss(const array &input, float azimuth, float elevation, float depth)
{
    if (depth<1 || depth>100) {
        printf("Depth should be in the range of 1-100");
        return input;
    }
    static float x[3] = { -1, 0, 1 };
    static array hg(3, x);
    static array vg = hg.T();
    array in = input;
    if (in.dims(2)>1)
        in = colorSpace(input, AF_GRAY, AF_RGB);
    else
        in = input;
    // convert angles to radians
    float phi = elevation*af::Pi / 180.0f;
    float theta = azimuth*af::Pi / 180.0f;
    // compute light pos in cartesian coordinates
    // and scale with maximum intensity
    // phi will effect the amount of we intend to put
    // on a pixel
    float pos[3];
    pos[0] = 255.99f * cos(phi)*cos(theta);
    pos[1] = 255.99f * cos(phi)*sin(theta);
    pos[2] = 255.99f * sin(phi);
    // compute gradient vector
    array gx = convolve(in, vg);
    array gy = convolve(in, hg);
    float pxlz = (6 * 255.0f) / depth;
    array zdepth = constant(pxlz, gx.dims());
    array vdot = gx*pos[0] + gy*pos[1] + pxlz*pos[2];
    array outwd = vdot < 0.0f;
    array norm = vdot / sqrt(gx*gx + gy*gy + zdepth*zdepth);
    array color = outwd * 0.0f + (1 - outwd) * norm;
    return color;
}
int main(int argc, char **argv)
{
    //try {
        int device = argc > 1 ? atoi(argv[1]) : 0;
        std::cout << device << std::endl;
        af::setDevice(device);
        af::info();

        clock_t t = clock();
        array img = loadImage("/home/kevin/test/test.jpg", true);
        
        float min = af::min<float>(img);
    	float max = af::max<float>(img);
    	std::cout << min << " " << max << std::endl;

        array mf = medianfilter(img, 15, 15);
        //array mf2 = medianfilter(mf, 15, 15);
        std::cout << "median filtering time: " << float(clock()-t)/CLOCKS_PER_SEC << std::endl;

        af::Window wnd("Image Filters Demo");
        std::cout << "Press ESC while the window is in focus to exit" << std::endl;
        while (!wnd.close()) {
        	wnd.grid(1,2);
        	//normalizeImage(img, min, max);
            wnd(0,1).image(img / 2000, "Original image");
        	//normalizeImage(mf, min, max);
            wnd(0,0).image(mf / 2000, "Median filter");
            wnd.show();
        }
    /*}
    catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }*/
    return 0;
}/*
* @Author: kevin
* @Date:   2016-12-15 18:20:03
* @Last Modified by:   kevin
* @Last Modified time: 2016-12-20 22:26:34
*/
