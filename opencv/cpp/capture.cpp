/*
   capture.cpp - Simple demo of OpenCV image capture using Leopard Imageing M021 camear on Linux.
  
   Copyright (C) 2016 Simon D. Levy

   This file is part of M021_V4L2.

   M021_V4L2 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   BreezySTM32 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with M021_V4L2.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>
#include <sys/timeb.h>

#include "m021_v4l2_opencv.hpp"

// http://codepad.org/qPsNtwzp
static int getMilliCount(void){

    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}

// http://www.codepool.biz/image-processing-opencv-gamma-correction.html
static void GammaCorrection(Mat& src, Mat& dst, float fGamma)
{

    unsigned char lut[256];

    for (int i = 0; i < 256; i++) {

        lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
    }

    dst = src.clone();

    MatIterator_<Vec3b> it, end;

    for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++) {
        (*it)[0] = lut[((*it)[0])];
        (*it)[1] = lut[((*it)[1])];
        (*it)[2] = lut[((*it)[2])];
    }
}

// http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
static void SimplestCB(Mat& in, Mat& out, float percent) {

    assert(in.channels() == 3);
    assert(percent > 0 && percent < 100);

    float half_percent = percent / 200.0f;

    vector<Mat> tmpsplit; split(in,tmpsplit);

    for(int i=0;i<3;i++) {

        //find the low and high precentile values (based on the input percentile)
        Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
        cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
        int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));

        //saturate below the low percentile and above the high percentile
        tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
        tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

        //scale the channel
        normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
    }

    merge(tmpsplit,out);
}

int main()
{
    Mat src;
    Mat dst1;
    Mat dst2;

    M021_800x460_Capture cap(src);

    int start = getMilliCount();

    while (true) {

        SimplestCB(src, dst1, 25);

        GammaCorrection(dst1, dst2, 0.95);

        imshow("LI-USB30-M021", dst2);

        if (cvWaitKey(1) == 27) 
            break;
    }

    double duration = (getMilliCount() - start) / 1000.;

    int count = cap.getCount();

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    return 0;
}
