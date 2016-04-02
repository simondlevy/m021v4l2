#include <opencv2/core/core.hpp>

#include "m021_v4l2.h"

class M021_800x460_Capture {

    private:

        m021_800x460_t cap;

    public:

        M021_800x460_Capture(int id) {

            m021_800x460_init(id, &cap);
        }

        void grab(Mat & mat) {

            m021_800x460_grab_bgr(&this->cap, mat.data);
        }
};
