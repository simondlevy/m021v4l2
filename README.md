# M021V4L2: Capture images from Leopard Imaging LI-USB30-M021 camera on Linux in C++ and Python

The LI-USB30-M021 camera from Leopard Imaging is a fast (up to 90 frames per
second) global-shutter camera that captures images over USB 3.0.  Because the
camera serves up raw image bytes, getting it to work on Linux using V4L2 (Video
For Linux 2) requires a bit of extra format-conversion work.  With some help
from the folks at Leopard Imaging, I was able to write a few simple APIs for
the camera for people who want to use it on Linux without doing the conversion
themselves.

The C++ and Python APIs are intended for OpenCV users who want to be able to
capture images as a Mat object (C++) or NumPy array (Python).  Because the M021
camera supports three frame sizes (1280 x 720 at 60 FPS; 800 x 460 at 90 FPS;
640x480 at 30 FPS), I've provided a class for each frame size. The capture runs
on its own automatically-launched thread. As the code fragment below shows,
the classes are extremely simple to use:

<pre>
    cap = Capture800x460()

    while True:

        ret, frame = cap.read()

        cv2.imshow('LI-USB30-M021',frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
 </pre>

To run the OpenCV capture demo, cd to <b>opencv/cpp</b> and type <b>make
run</b>.  This program does some post-capture color balancing to compensate for
the slightly dimmed / green appearance of the image from the camera.

For your own OpenCV projects, I would put the C and C++ files
(<tt>m021\_v4l2.h</tt>, <tt>m021\_v4l2.c</tt>, <tt>m021\_v4l2\_opencv.hpp</tt>,
<tt>m021\_v4l2\_opencv.hpp</tt>) and modify the Makefile appropriately.

I've also provided a C API (which is used by the C++ and Python code) for
capturing images in YUYV format, along with a demo program (a cut-down version
of Guvcview) that displays the images using GTK and SDL.  To run the GTK/SDL
demo, cd to <b>gtksdl</b> and type <b>make run</b>.
