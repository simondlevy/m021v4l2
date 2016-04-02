#include <opencv2/core/core.hpp>

#include "m021_v4l2.h"

#include <stdio.h>
#include <stdlib.h>

class M021_800x460_Capture {

    private:

        m021_800x460_t cap;

    public:

        M021_800x460_Capture(int id);

        void grab(Mat & mat);
};
