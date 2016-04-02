#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>
#include <sys/timeb.h>

#include "m021_opencv.hpp"

static int getMilliCount(){

    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}

int main()
{
    Mat mat;

    m021_800x460_capture(mat);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    int start = getMilliCount();

    while (true) {

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27) 
            break;

    }

    double duration = (getMilliCount() - start) / 1000.;

    int count = m021_800x460_getcount();

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    return 0;
}
