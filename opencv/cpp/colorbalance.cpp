#include "colorbalance.hpp"
#include <stdio.h>

// http://www.morethantechnical.com/2015/01/14/simplest-color-balance-with-opencv-wcode/
void ColorBalance(Mat& src, Mat& dst, float percent) {

    assert(src.channels() == 3);
    assert(percent > 0 && percent < 100);

    float half_percent = percent / 200.0f;

    vector<Mat> tmpsplit; split(src, tmpsplit);

    int lows[3]  = {18, 18,  18};
    int highs[3] = {75, 142, 125};

    for(int i=0;i<3;i++) {

        //find the low and high precentile values (based on the input percentile)
        Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
        cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        int lowval = lows[i]; //flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
        int highval = highs[i]; //flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));

        //saturate below the low percentile and above the high percentile
        tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
        tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

        //scale the channel
        normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
    }

    printf("\n");

    merge(tmpsplit, dst);
}
