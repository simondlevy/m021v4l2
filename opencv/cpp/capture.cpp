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


pthread_mutex_t lock;

static void * main_loop(void * arg)
{
    Mat * mat = (Mat *)arg;

    M021_800x460_Capture cap(0);

    int count = 0;
    int start = getMilliCount();

    while (true) {

        pthread_mutex_lock(&lock);

        cap.grab(*mat);

        pthread_mutex_unlock(&lock);

        count++;

    }

    double duration = (getMilliCount() - start) / 1000.;

    printf("%d frames in %3.3f seconds = %3.3f frames /sec \n", count, duration, count/duration);

    return (void *)0;
}

int main()
{
    Mat mat;

    if (pthread_mutex_init(&lock, NULL) != 0)
    {
        printf("\n mutex init failed\n");
        return 1;
    }

    if (pthread_create(&video_thread, NULL, main_loop, &mat)) {
        fprintf(stderr, "Failed to create thread\n");
        exit(1);
    }

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    while (true) {

        if (mat.data)
            imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27) 
            break;

    }

    pthread_join(video_thread, NULL);

    return 0;
}
