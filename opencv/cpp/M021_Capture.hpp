#include <opencv2/core/core.hpp>

#include "m021_v4l2.h"

#include <stdio.h>
#include <stdlib.h>

class M021_1280x720_Capture {

    private:

        m021_1280x720_t cap;

    public:

        M021_1280x720_Capture(int id) {

            m021_1280x720_init(id, &cap);
        }

        void grab(Mat & mat) {

            if (!mat.data) {
                mat = Mat(720, 1280, CV_8UC3);
            }

            m021_1280x720_grab_bgr(&this->cap, mat.data);
        }
};

class M021_800x460_Capture {

    private:

        m021_800x460_t cap;

    public:

        M021_800x460_Capture(int id) {

            m021_800x460_init(id, &cap);
        }

        void grab(Mat & mat) {

            if (!mat.data) {
                mat = Mat(460, 800, CV_8UC3);
            }

            m021_800x460_grab_bgr(&this->cap, mat.data);
        }
};
