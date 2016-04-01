#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <stdio.h>
#include <sys/timeb.h>

#include "../../m021_v4l2.h"

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

    Mat mat(460, 800, CV_8UC3);

    //vdIn_t * cap = (vdIn_t *)malloc(sizeof(vdIn_t));
    vdIn_t cap;

    m021_init_800x460("/dev/video0", &cap);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    int count = 0;
    int start = getMilliCount();

    while (true) {

        m021_grab_bgr(&cap, mat.data);

        mat *= 1.5;

        count++;

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27) 
            break;
    }

    //m021_free(cap);

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
