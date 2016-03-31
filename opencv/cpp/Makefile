LIBS = -L /usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lpthread -lv4l2 -ludev

capture: capture.cpp ../../m021_v4l2.c ../../m021_v4l2.h
	g++ -Wall -o capture capture.cpp ../../m021_v4l2.c $(LIBS)

run: capture
	./capture

clean:
	rm -f capture *~ 

commit:
	git commit -a --allow-empty-message -m '' 
	git push
