#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

void m021_800x460_capture(Mat & mat);

int m021_800x460_getcount(void);

class M021_800x460_Capture {

    public:

        M021_800x460_Capture(Mat & mat);

        int getCount(void);

    private: 

        pthread_t video_thread;
        pthread_mutex_t lock;

        int count;
};

