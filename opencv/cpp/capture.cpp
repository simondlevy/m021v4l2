#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include "../../m021_v4l2.h"

int main()
{
    Mat mat(460, 800, CV_8UC3);

    uint8_t buf[800*460*3];

    vdIn_800x460_t cap;
    m021_init_800x460("/dev/video0", &cap);

    cvNamedWindow("window",CV_WINDOW_AUTOSIZE);

    while (true) {

        m021_grab_800x460_bgr(&cap, buf);

        memcpy(mat.data, buf, 460*800*3);

        imshow("window", mat);

        if (cvWaitKey(1) == 27)
            break;
    }

    return 0;
}
