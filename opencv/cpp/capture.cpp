#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>

#include "../../m021_v4l2.h"

int main()
{
    // XXX not sure why we have to do this!
    uint8_t dummy[4000];

    Mat mat(480, 640, CV_8UC3);

    vdIn_640x480_t cap;

    m021_640x480_init("/dev/video0", &cap);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    int count = 0;

    while (true) {

        m021_640x480_grab_bgr(&cap, mat.data);

        count++;

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27)
            break;
    }

    printf("%d\n", count);

    return 0;
}
