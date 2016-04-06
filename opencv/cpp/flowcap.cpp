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
    Mat prev, next; 

    VideoCapture cap(0); 

    static const int SCALEDOWN = 1;

    while (true) {   

        cap >> img;

        resize(img, next, Size(img.size().width>>SCALEDOWN, img.size().height>>SCALEDOWN) );

        cvtColor(next, next, CV_BGR2GRAY);

        if (prev.data) {

            Mat flow;
            calcOpticalFlowFarneback(prev, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

            drawOptFlowMap(flow, next, 10, CV_RGB(0, 255, 0));
        }

        imshow("Image", next);

        if (waitKey(5) >= 0)   
            break;

        prev = next;
    }
}



