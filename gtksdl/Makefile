CC = gcc

guvcview_OBJECTS = \
	guvcview-close.o \
	guvcview-globals.o \
	guvcview-guvcview.o \
	guvcview-ms_time.o \
	guvcview-timers.o \
	guvcview-m021_v4l2.o \
	guvcview-video.o

guvcview_LINK = $(CC) $(guvcview_CFLAGS) $(CFLAGS) \
	$(guvcview_LDFLAGS) $(LDFLAGS) -o $@

QUIET = @echo "  CC      " $@;

COMPILE = $(CC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) \
	$(CPPFLAGS) $(AM_CFLAGS) $(CFLAGS)

LINK = $(CC) $(AM_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@

INCLUDES = -I/usr/include/gtk-3.0 \
		   -I/usr/include/pango-1.0 \
		   -I/usr/include/atk-1.0 \
		   -I/usr/include/cairo \
		   -I/usr/include/gdk-pixbuf-2.0 \
		   -I/usr/include/glib-2.0 \
		   -I/usr/lib/x86_64-linux-gnu/glib-2.0/include

DEPS_CFLAGS = $(INCLUDES) -pthread -D_GNU_SOURCE=1 -D_REENTRANT -pthread 

DEPS_LIBS = -lgtk-3 -lgdk-3 -lpango-1.0 -lgobject-2.0 -lglib-2.0 -lSDL -lpthread -lv4l2 -ludev

guvcview_SOURCES = \
	close.h \
	close.c \
	defs.h \
	globals.h \
	globals.c \
	guvcview.h \
	guvcview.c \
	jdatatype.h \
	ms_time.h \
	ms_time.c \
	timers.h \
	timers.c \
	../m021_v4l2.h \
	../m021_v4l2.c \
	video.h \
	video.c

guvcview_CFLAGS = -DPACKAGE_LOCALE_DIR=\""$(prefix)/$(DATADIRNAME)/locale"\" \
				  -DPACKAGE_SRC_DIR=\""$(srcdir)"\" \
				  -DPACKAGE_DATA_DIR=\""$(datadir)"\" \
				  $(DEPS_CFLAGS) \
				  $(EXTRA_CFLAGS) \
				  $(PTHREAD_CFLAGS) \
				  -D_REENTRANT \
	 			  -D_FILE_OFFSET_BITS=64 \
	 			  -Wall

guvcview_LDFLAGS = $(LIBINTL)
guvcview_LDADD = $(DEPS_LIBS) $(EXTRA_LIBS) $(PTHREAD_LIBS)

all: guvcview

guvcview: $(guvcview_OBJECTS) $(guvcview_DEPENDENCIES) $(EXTRA_guvcview_DEPENDENCIES) 
	$(guvcview_LINK) $(guvcview_OBJECTS) $(guvcview_LDADD) $(LIBS)

guvcview-close.o: close.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-close.o close.c

guvcview-globals.o: globals.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-globals.o globals.c

guvcview-guvcview.o: guvcview.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-guvcview.o guvcview.c

guvcview-ms_time.o: ms_time.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-ms_time.o ms_time.c

guvcview-timers.o: timers.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-timers.o timers.c

guvcview-m021_v4l2.o: ../m021_v4l2.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-m021_v4l2.o ../m021_v4l2.c

guvcview-video.o: video.c *.h
	$(QUIET) $(CC)  $(guvcview_CFLAGS) -c -o guvcview-video.o video.c

clean:
	rm -f guvcview *.o

run: guvcview
	./guvcview

commit:
	git commit -a --allow-empty-message -m ''; git push

count:
	wc -l *.c | sort -nr
