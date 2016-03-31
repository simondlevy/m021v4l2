#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include "../../m021_v4l2.h"

int main()
{
    Mat mat(460, 800, CV_8UC3);

    uint8_t buf[800*460*3];

    vdIn_800x460_t cap;
    m021_800x460_init("/dev/video0", &cap);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    while (true) {

        //m021_800x460_grab_bgr(&cap, buf);
        m021_800x460_grab_bgr(&cap, mat.data);

        //memcpy(mat.data, buf, 460*800*3);

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27)
            break;
    }

    return 0;
}
