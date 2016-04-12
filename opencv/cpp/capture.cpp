/*
   capture.cpp - Simple demo of OpenCV image capture using Leopard Imageing M021 camera on Linux.
  
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
#include "colorbalance.hpp"

static const float COLORBALANCE = 0.5;

// http://codepad.org/qPsNtwzp
static int getMilliCount(void){

    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}

int main()
{
    Mat img;
    Mat img2;

    M021_800x460_Capture cap(img);

    int start = getMilliCount();

    int count = 0;

    while (true) {

        ColorBalance(img, img2, COLORBALANCE);
        
        imshow("LI-USB30-M021", img2);

        if (cvWaitKey(1) == 27) 
            break;

        count++;
    }

    double duration = (getMilliCount() - start) / 1000.;

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    return 0;
}
