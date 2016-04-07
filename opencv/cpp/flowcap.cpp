#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#include "m021_v4l2_opencv.hpp"
#include "colorbalance.hpp"

static const float COLORBALANCE = 0.5;

static const double PYRSCALE   = 0.5;
static const int    LEVELS     = 3;
static const int    WINSIZE    = 15;
static const int    ITERATIONS = 3;
static const int    POLY_N     = 5;
static const double POLY_SIGMA = 1.2;

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv)
{
    //VideoCapture cap(0);

    Mat flow, cflow, frame, bright;
    Mat gray, prevgray, uflow;
    
    M021_800x460_Capture cap(frame);

    for(;;)
    {
        //cap >> frame;
        ColorBalance(frame, bright, COLORBALANCE);

        cvtColor(bright, gray, COLOR_BGR2GRAY);

        if( !prevgray.empty() )
        {
            calcOpticalFlowFarneback(prevgray, gray, uflow, 
                PYRSCALE, LEVELS, WINSIZE, ITERATIONS, POLY_N, POLY_SIGMA, 0);
            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
            uflow.copyTo(flow);
            drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
            imshow("flow", cflow);
        }

        if(waitKey(1)>=0)
            break;

        std::swap(prevgray, gray);
    }
    return 0;
}
