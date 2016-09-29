# Makefile : builds M021_V4L2 OpenCV examples
#
# Copyright (C) Simon D. Levy 2016
#
# This file is part of M021_V4L2.
#
# M021_V4L2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# M021_V4L2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with M021_V4L2.  If not, see <http://www.gnu.org/licenses/>.

LIBS = -L /usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video \
	   -lpthread -lv4l2 -ludev

all: capture flowcap

test: capture
	./capture

cap: capture
	./capture

capture: capture.cpp m021_v4l2_opencv.cpp m021_v4l2_opencv.hpp ../../m021_v4l2.c ../../m021_v4l2.h 
	g++ -I ../.. -Wall -o capture capture.cpp m021_v4l2_opencv.cpp ../../m021_v4l2.c $(LIBS)

flow: flowcap
	./flowcap

flowcap: flowcap.cpp m021_v4l2_opencv.cpp m021_v4l2_opencv.hpp ../../m021_v4l2.c ../../m021_v4l2.h 
	g++ -I ../.. -Wall -o flowcap flowcap.cpp m021_v4l2_opencv.cpp ../../m021_v4l2.c $(LIBS)

clean:
	rm -rf capture flowcap Documentation *~ 

edit:
	vim capture.cpp

doc: Doxyfile
	doxygen

commit:
	git commit -a --allow-empty-message -m '' 
	git push
