'''
m021v4l2 : Python classes for Leopard Imaging LI-USB30-M021 on Linux

Copyright (C) 2016 Simon D. Levy

This file is part of M021_V4L2.

M021_V4L2 is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
BreezySTM32 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with M021_V4L2.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import libm021v4l2 as lib

# XXX This is kind of a sleazy implementation, partly in Python, partly in C extension.  
# XXX We probably should do the whole thing in C.

class _Capture(object):

    def __init__(self, rows, cols, bcorrect, gcorrect, rcorrect):

        self.frame = np.zeros((rows,cols,3), dtype='uint8')

        lib.init(self.frame, bcorrect, gcorrect, rcorrect) 

    def read(self):

        lib.acquire(self.frame)

        return True, self.frame

    def getCount(self):

        return lib.count()

class Capture1280x720(_Capture):

    def __init__(self, bcorrect=50, gcorrect=0, rcorrect=50):

        _Capture.__init__(self, 720, 1280, bcorrect, gcorrect, rcorrect) 

class Capture800x460(_Capture):

    def __init__(self, bcorrect=50, gcorrect=0, rcorrect=50):

        _Capture.__init__(self, 460, 800, bcorrect, gcorrect, rcorrect) 

class Capture640x480(_Capture):

    def __init__(self, bcorrect=50, gcorrect=0, rcorrect=50):

        _Capture.__init__(self, 480, 640, bcorrect, gcorrect, rcorrect) 


