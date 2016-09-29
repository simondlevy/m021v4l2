#!/usr/bin/env python

import libm021v4l2
import numpy as np

a = np.zeros((8, 7, 3, 2))
b = np.array([[1.1, 2.2, 3.3], [1.2, 1.3, 1.4]])
c = np.array([[1.1, 2.2, 3.3], [1.2, 1.3, 1.4]])
libm021v4l2.example(a, b, c)
print(c)
