#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace cv;
using namespace std;

void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
 for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at< Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
        }
    }


int main()
{
    Mat img;
    Mat prvs, next; 

    VideoCapture cap(0); 

    int s = 2;

    while (true) {   

        cap >> img;

        resize(img, next, Size(img.size().width/s, img.size().height/s) );
        cvtColor(next, next, CV_BGR2GRAY);

       /* 
        Mat flow;
        calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        Mat cflow;
        cvtColor(prvs, cflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 10, CV_RGB(0, 255, 0));
        imshow("OpticalFlowFarneback", cflow);

        imshow("prvs", prvs);
        imshow("next", next);
        */

        imshow("Image", img);

        if (waitKey(5) >= 0)   
            break;

        //prvs = next.clone();
    }
}


