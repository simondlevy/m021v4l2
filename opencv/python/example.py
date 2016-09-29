#!/usr/bin/env python

import libm021v4l2
import numpy as np

c = np.array([[1.1, 2.2, 3.3], [1.2, 1.3, 1.4]])

libm021v4l2.acquire(c)

print(c)
