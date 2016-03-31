#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>
#include <sys/timeb.h>

#include "../../m021_v4l2.h"

// http://www.firstobject.com/getmillicount-milliseconds-portable-c++.htm
static int getMilliCount(){
    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}


int main()
{
    // XXX not sure why we have to do this!
    uint8_t dummy[4000];

    Mat mat(460, 800, CV_8UC3);

    vdIn_800x460_t cap;

    m021_800x460_init("/dev/video0", &cap);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    int count = 0;
    int start = getMilliCount();

    while (true) {

        m021_800x460_grab_bgr(&cap, mat.data);

        mat *= 1.5;

        count++;

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27)
            break;
    }

    double duration = (getMilliCount() - start) / 1000.;

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    return 0;
}
