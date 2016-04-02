#include "m021_v4l2.h"
#include "m021_opencv.hpp"

#include <stdio.h>

typedef struct {

    Mat mat;
    pthread_mutex_t lock;
    int count;

} data_t;


static void * loop(void * arg)
{

    m021_800x460_t cap;
    m021_800x460_init(0, &cap);

    data_t * data = (data_t *)arg;
    pthread_mutex_t lock = data->lock;
    Mat mat = data->mat;

    data->count = 0;

    while (true) {

        pthread_mutex_lock(&lock);

        m021_800x460_grab_bgr(&cap, mat.data);

        pthread_mutex_unlock(&lock);

        data->count++;
    }

    return (void *)0;
}

M021_800x460_Capture::M021_800x460_Capture(Mat & mat) {

    mat = Mat(460, 800, CV_8UC3);

    pthread_mutex_t lock;

    if (pthread_mutex_init(&lock, NULL) != 0) {
        printf("\n mutex init failed\n");
        exit(1);
    }

    data_t * data = new data_t;
    data->mat = mat;
    data->lock = lock;

    this->data = data;

    if (pthread_create(&this->video_thread, NULL, loop, data)) {
        fprintf(stderr, "Failed to create thread\n");
        exit(1);
    }
}
        
M021_800x460_Capture::~M021_800x460_Capture(void)
{
    printf("DESTROY\n");
}

int M021_800x460_Capture::getCount(void) 
{
    data_t * data = (data_t *)this->data;

    return data->count;
}
