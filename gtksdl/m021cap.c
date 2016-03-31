#include <gtk/gtk.h>
#include <SDL/SDL.h>
#include <glib/gi18n.h>
#include <glib/gprintf.h>
#include <fcntl.h>		

#include "../m021_v4l2.h"

#define __THREAD_TYPE pthread_t
#define __THREAD_CREATE(t,f,d) (pthread_create(t,NULL,f,d))
#define __THREAD_JOIN(t) (pthread_join(t, NULL))

#define __MUTEX_TYPE pthread_mutex_t
#define __INIT_MUTEX(m) ( pthread_mutex_init(m, NULL) )
#define __GMUTEX &mutex

#define MEDIUM

#ifdef SMALL
#define VDIN_T vdIn_640x480_t
#define WIDTH 640
#define HEIGHT 480
#define VD_INIT m021_init_640x480
#define VD_GRAB m021_grab_640x480_yuyv
#else
#ifdef MEDIUM
#define VDIN_T vdIn_800x460_t
#define WIDTH 800
#define HEIGHT 460
#define VD_INIT m021_init_800x460
#define VD_GRAB m021_grab_800x460_yuyv
#else
#define VDIN_T vdIn_1280x720_t
#define WIDTH 1280
#define HEIGHT 720
#define VD_INIT m021_init_1280x720
#define VD_GRAB m021_grab_1280x720_yuyv
#endif
#endif


typedef char * pchar;

#define WINSIZEX 560
#define WINSIZEY 560

#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480

static	__MUTEX_TYPE mutex;      //global struct mutex

static int hwaccel;             //use hardware acceleration
static int bpp;                 //current bytes per pixel
static char *caption;           //title bar caption
static gboolean signalquit;
static	int desktop_w;          //Desktop width
static	int desktop_h;          //Desktop height
static	int winwidth;           //control windoe width
static	int winheight;          //control window height
static	int framewidth;         //frame width
static	int frameheight;        //frame height

static VDIN_T *videoIn;
static struct GWIDGET *gwidget;
static __THREAD_TYPE video_thread;
static Uint32 SDL_VIDEO_Flags = SDL_ANYFORMAT | SDL_RESIZABLE;
static const SDL_VideoInfo *info;

static int initGlobals (void)
{
	__INIT_MUTEX( __GMUTEX );

	caption = g_new(char, 32);

	g_sprintf(caption,"LI-USB30-M021");

	bpp = 0; //current bytes per pixel
	hwaccel = 1; //use hardware acceleration
	desktop_w = 0;
	desktop_h = 0;
	framewidth = DEFAULT_WIDTH;
	frameheight = DEFAULT_HEIGHT;
	winwidth=WINSIZEX;
	winheight=WINSIZEY;

	/* reset with videoIn parameters */
	return (0);
}

/* Must set this as global so they */
/* can be set from any callback.   */

struct GWIDGET
{
	GMainLoop *main_loop;

	/* The main window*/
	GtkWidget *mainwin;

	gboolean vid_widget_state;
	int status_warning_id;
};

static void shutdown (void)
{
	/* wait for the video thread */
    signalquit = TRUE;
    __THREAD_JOIN(video_thread);

    gtk_window_get_size(GTK_WINDOW(gwidget->mainwin),&(winwidth),&(winheight));//mainwin or widget

	gtk_main_quit();

}
static int shutdown_timer(gpointer data)
{
    /*stop video capture*/
    shutdown ();
    
    return (FALSE);/*destroys the timer*/
}

static SDL_Overlay * video_init(SDL_Surface **pscreen)
{
    int width = framewidth;
    int height = frameheight;

    if (*pscreen == NULL) //init SDL
    {
        /*----------------------------- Test SDL capabilities ---------------------*/
        if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER) < 0)
        {
            g_printerr("Couldn't initialize SDL: %s\n", SDL_GetError());
            exit(1);
        }

        /* For this version, we will use hardware acceleration as default*/
        if(hwaccel)
        {
            if ( ! getenv("SDL_VIDEO_YUV_HWACCEL") ) putenv("SDL_VIDEO_YUV_HWACCEL=1");
            if ( ! getenv("SDL_VIDEO_YUV_DIRECT") ) putenv("SDL_VIDEO_YUV_DIRECT=1");
        }
        else
        {
            if ( ! getenv("SDL_VIDEO_YUV_HWACCEL") ) putenv("SDL_VIDEO_YUV_HWACCEL=0");
            if ( ! getenv("SDL_VIDEO_YUV_DIRECT") ) putenv("SDL_VIDEO_YUV_DIRECT=0");
        }

        info = SDL_GetVideoInfo();

        if (info->hw_available)
        {
            SDL_VIDEO_Flags |= SDL_HWSURFACE;
            SDL_VIDEO_Flags |= SDL_DOUBLEBUF;
        }
        else
        {
            SDL_VIDEO_Flags |= SDL_SWSURFACE;
        }

        if (info->blit_hw)
        {
            SDL_VIDEO_Flags |= SDL_ASYNCBLIT;
        }

        if(!desktop_w) desktop_w = info->current_w; //get desktop width
        if(!desktop_h) desktop_h = info->current_h; //get desktop height

        SDL_WM_SetCaption(caption, NULL);

        /* enable key repeat */
        SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY,SDL_DEFAULT_REPEAT_INTERVAL);
    }
    /*------------------------------ SDL init video ---------------------*/

    g_print("Checking video mode %ix%i@32bpp : ", width, height);
    int bpp = SDL_VideoModeOK(
        width,
        height,
        32,
        SDL_VIDEO_Flags);

    if(!bpp)
    {
        g_print("Not available \n");
        /*resize video mode*/
        if ((width > desktop_w) || (height > desktop_h))
        {
            width = desktop_w; /*use desktop video resolution*/
            height = desktop_h;
        }
        else
        {
            width = 800;
            height = 600;
        }
        g_print("Resizing to %ix%i\n", width, height);

    }
    else
    {
        g_print("OK \n");
        bpp = bpp;
    }

    *pscreen = SDL_SetVideoMode(
        width,
        height,
        bpp,
        SDL_VIDEO_Flags);

    if(*pscreen == NULL)
    {
        return (NULL);
    }
    //use requested resolution for overlay even if not available as video mode
    SDL_Overlay* overlay=NULL;
    overlay = SDL_CreateYUVOverlay(framewidth, frameheight,
        SDL_YUY2_OVERLAY, *pscreen);

    SDL_ShowCursor(SDL_DISABLE);
    return (overlay);
}

/*-------------------------------- Main Video Loop ---------------------------*/
/* run in a thread (SDL overlay)*/
static void *main_loop()
{
    struct particle* particles = NULL; //for the particles video effect

    SDL_Event event;
    /*the main SDL surface*/
    SDL_Surface *pscreen = NULL;
    SDL_Overlay *overlay = NULL;
    SDL_Rect drect;

    uint8_t *p = NULL;

    signalquit = FALSE;

    overlay = video_init(&(pscreen));

    if(overlay == NULL)
    {
        g_print("FATAL: Couldn't create yuv overlay - please disable hardware accelaration\n");
        signalquit = TRUE; /*exit video thread*/
    }
    else
    {
        p = (unsigned char *) overlay->pixels[0];

        drect.x = 0;
        drect.y = 0;
        drect.w = pscreen->w;
        drect.h = pscreen->h;
    }

    while (!signalquit)
    {
        SDL_LockYUVOverlay(overlay);
        if (VD_GRAB(videoIn, p) < 0) {
            g_printerr("Error grabbing image \n");
            continue;
        }
        SDL_UnlockYUVOverlay(overlay);
        SDL_DisplayYUVOverlay(overlay, &drect);

        while( SDL_PollEvent(&event) )
        {
            //printf("event type:%i  event key:%i\n", event.type, event.key.keysym.scancode);
            if(event.type==SDL_VIDEORESIZE)
            {
                pscreen = SDL_SetVideoMode(event.resize.w, event.resize.h, bpp, SDL_VIDEO_Flags);
                drect.w = event.resize.w;
                drect.h = event.resize.h;
            }
            if(event.type==SDL_QUIT)
            {
                //shutDown
                g_timeout_add(200, shutdown_timer, NULL);
            }
        }

    }/*loop end*/


    p = NULL;
    if(particles) g_free(particles);
    particles=NULL;

    fflush(NULL);//flush all output buffers

    if(overlay)
        SDL_FreeYUVOverlay(overlay);

    SDL_Quit();


    videoIn = NULL;
    return ((void *) 0);
}

/*
 * Unix signals that are cought are written to a pipe. The pipe connects
 * the unix signal handler with GTK's event loop. The array signal_pipe will
 * hold the file descriptors for the two ends of the pipe (index 0 for
 * reading, 1 for writing).
 */
static int signal_pipe[2];

/*
 * The unix signal handler.
 * Write any unix signal into the pipe. The writing end of the pipe is in
 * non-blocking mode. If it is full (which can only happen when the
 * event loop stops working) signals will be dropped.
 */
static void pipe_signals(int signal)
{
  if(write(signal_pipe[1], &signal, sizeof(int)) != sizeof(int))
    {
      fprintf(stderr, "unix signal %d lost\n", signal);
    }
}

/*
 * The event loop callback that handles the unix signals. Must be a GIOFunc.
 * The source is the reading end of our pipe, cond is one of
 *   G_IO_IN or G_IO_PRI (I don't know what could lead to G_IO_PRI)
 * the pointer d is always NULL
 */
static gboolean deliver_signal(GIOChannel *source, GIOCondition cond, gpointer data)
{
  GError *error = NULL;		/* for error handling */

  /*
   * There is no g_io_channel_read or g_io_channel_read_int, so we read
   * char's and use a union to recover the unix signal number.
   */
  union {
    gchar chars[sizeof(int)];
    int signal;
  } buf;
  GIOStatus status;		/* save the reading status */
  gsize bytes_read;		/* save the number of chars read */

  /*
   * Read from the pipe as long as data is available. The reading end is
   * also in non-blocking mode, so if we have consumed all unix signals,
   * the read returns G_IO_STATUS_AGAIN.
   */
  while((status = g_io_channel_read_chars(source, buf.chars,
		     sizeof(int), &bytes_read, &error)) == G_IO_STATUS_NORMAL)
    {
      g_assert(error == NULL);	/* no error if reading returns normal */

      /*
       * There might be some problem resulting in too few char's read.
       * Check it.
       */
      if(bytes_read != sizeof(int)){
	fprintf(stderr, "lost data in signal pipe (expected %lu, received %lu)\n",
		(long unsigned int) sizeof(int), (long unsigned int) bytes_read);
	continue;	      /* discard the garbage and keep fingers crossed */
      }

      /* Ok, we read a unix signal number, so let the label reflect it! */
     switch (buf.signal)
     {
     	case SIGINT:
     		shutdown();
     		break;
    	default:
    		printf("guvcview signal %d caught\n", buf.signal);
    		break;
     }
    }

  /*
   * Reading from the pipe has not returned with normal status. Check for
   * potential errors and return from the callback.
   */
  if(error != NULL){
    fprintf(stderr, "reading signal pipe failed: %s\n", error->message);
    exit(1);
  }
  if(status == G_IO_STATUS_EOF){
    fprintf(stderr, "signal pipe has been closed\n");
    exit(1);
  }

  g_assert(status == G_IO_STATUS_AGAIN);
  return (TRUE);		/* keep the event source */
}

/*--------------------------------- MAIN -------------------------------------*/
int main(int argc, char *argv[])
{
	/*
   	* In order to register the reading end of the pipe with the event loop
   	* we must convert it into a GIOChannel.
   	*/
  	GIOChannel *g_signal_in;
  	long fd_flags; 	    /* used to change the pipe into non-blocking mode */
  	GError *error = NULL;	/* handle errors */

	initGlobals();

	/*---------------------------------- Allocations -------------------------*/

	gwidget = g_new0(struct GWIDGET, 1);
	gwidget->vid_widget_state = TRUE;

    if(!gtk_init_check(&argc, &argv))
    {
        g_printerr("can't open display\n");
        exit(1);
    }

    g_set_application_name(_("LEOPARD Video Capture"));

    /* make sure the type is realized so that we can change the properties*/
    g_type_class_unref (g_type_class_ref (GTK_TYPE_BUTTON));
    /* make sure gtk-button-images property is set to true (defaults to false in karmic)*/
    g_object_set (gtk_settings_get_default (), "gtk-button-images", TRUE, NULL);

    //get screen resolution
    if((!desktop_w) || (!desktop_h))
    {
        GdkScreen* screen = NULL;
        desktop_w = gdk_screen_get_width(screen);
        desktop_h = gdk_screen_get_height(screen);
    }

    if((winwidth > desktop_w) && (desktop_w > 0))
        winwidth = desktop_w;
    if((winheight > desktop_h) && (desktop_h > 0))
        winheight = desktop_h;


    /*----------------------- init videoIn structure --------------------------*/
    videoIn = g_new0(VDIN_T, 1);

    framewidth =  WIDTH;
    frameheight = HEIGHT;

    VD_INIT("/dev/video0", videoIn);

    if( __THREAD_CREATE(&video_thread, main_loop, NULL))
    {
        g_printerr("Video thread creation failed\n");

    }

    /*
     * Set the unix signal handling up.
     * First create a pipe.
     */
    if(pipe(signal_pipe))
    {
        perror("pipe");
        exit(1);
    }

    /*
     * put the write end of the pipe into nonblocking mode,
     * need to read the flags first, otherwise we would clear other flags too.
     */
    fd_flags = fcntl(signal_pipe[1], F_GETFL);
    if(fd_flags == -1)
    {
        perror("read descriptor flags");
    }
    if(fcntl(signal_pipe[1], F_SETFL, fd_flags | O_NONBLOCK) == -1)
    {
        perror("write descriptor flags");
    }

    /* Install the unix signal handler pipe_signals for the signals of interest */
    signal(SIGINT, pipe_signals);
    signal(SIGUSR1, pipe_signals);
    signal(SIGUSR2, pipe_signals);

    /* convert the reading end of the pipe into a GIOChannel */
    g_signal_in = g_io_channel_unix_new(signal_pipe[0]);

    /*
     * we only read raw binary data from the pipe,
     * therefore clear any encoding on the channel
     */
    g_io_channel_set_encoding(g_signal_in, NULL, &error);
    if(error != NULL)
    {
        /* handle potential errors */
        fprintf(stderr, "g_io_channel_set_encoding failed %s\n",
                error->message);
    }

    /* put the reading end also into non-blocking mode */
    g_io_channel_set_flags(g_signal_in,
            g_io_channel_get_flags(g_signal_in) | G_IO_FLAG_NONBLOCK, &error);

    if(error != NULL)
    {		/* tread errors */
        fprintf(stderr, "g_io_set_flags failed %s\n",
                error->message);
    }

    /* register the reading end with the event loop */
    g_io_add_watch(g_signal_in, G_IO_IN | G_IO_PRI, deliver_signal, NULL);


    gtk_main();

    free(gwidget);

    g_print("Closing GTK... OK\n");
    return 0;
}
