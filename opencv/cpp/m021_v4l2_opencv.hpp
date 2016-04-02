#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

class M021_800x460_Capture {

    public:

        M021_800x460_Capture(Mat & mat);

        ~M021_800x460_Capture(void);

        unsigned long long getCount(void);

    private: 

        void * data;

        pthread_t video_thread;
};

