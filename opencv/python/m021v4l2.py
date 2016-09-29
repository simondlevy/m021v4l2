import numpy as np
import libm021v4l2 as lib

class Capture1280x720:

    def __init__(self):

        return

    def read(self):

        frame = np.zeros((720,1280,3), dtype='uint8')

        lib.acquire(frame)

        return True, frame

    def release(self):

        return
