#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>
#include <sys/timeb.h>

#include "M021_Capture.hpp"

static pthread_t video_thread;

// http://www.firstobject.com/getmillicount-milliseconds-portable-c++.htm
static int getMilliCount(){
    timeb tb;
    ftime(&tb);
    int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
    return nCount;
}

static void * main_loop(void * arg)
{
    bool * quit = (bool *)arg;

    Mat mat;

    M021_1280x720_Capture cap(0);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    int count = 0;
    int start = getMilliCount();

    while (true) {

        cap.grab(mat);

        count++;

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27) 
            break;
    }

    double duration = (getMilliCount() - start) / 1000.;

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    *quit = true;

    return (void *)0;
}

int main()
{
    bool quit = false;

    if(pthread_create(&video_thread, NULL, main_loop, &quit)) {
        fprintf(stderr, "Failed to create thread\n");
        exit(1);
    }

    while (!quit)
        ;
    
   return 0;
}
