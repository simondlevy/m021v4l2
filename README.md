# M021V4L2: Capture images from Leopard Imaging LI-USB30-M021 camera on Linux

The LI-USB30-M021 camera from Leopard Imaging is a fast (up to 90 frames per second) camera that captures
images over USB 3.0.  Because the camera serves up raw image bytes, getting it to work on Linux using V4L2 
(Video For Linux 2) requires a bit of extra format-conversion work.  With some help from the folks at Leopard Imaging, 
I was able to write a simple APIs for the camera for people who want to use it on Linux without doing the conversion
themselves.

The C++ API is intended for OpenCV users who want to be able to capture images in Mat format.  Because the M021
camera supports three frame sizes (1280 x 720 at 60 FPS; 800 x 460 at 90 FPS; 640x480 at 30 FPS), I've provided
a class for each frame size. The capture runs on its own automatically-launched thread. As the code fragment below 
shows,  the classes are extremely simple to use:

<pre>
    Mat mat;

    M021_800x460_Capture cap(mat);

    cvNamedWindow("LI-USB30-M021", CV_WINDOW_AUTOSIZE);

    while (true) {

        imshow("LI-USB30-M021", mat);

        if (cvWaitKey(1) == 27) 
            break;

    }

 </pre>

To run the OpenCV capture demo, cd to opencv/cpp and type <b>make run</b>.  To run the GTK/SDL demo, cd to 
gtksdl and type <b>make run</b>.

I've also provided a C API (which is used by my C++ code) for capturing images in YUYV format, along with a demo 
program (a cut-down version of Guvcview) that displays the images using GTK and SDL.  I'm working on a Python
API as well.
