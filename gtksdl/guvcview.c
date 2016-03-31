/*******************************************************************************#
#           guvcview              http://guvcview.sourceforge.net               #
#                                                                               #
#           Paulo Assis <pj.assis@gmail.com>                                    #
#           Nobuhiro Iwamatsu <iwamatsu@nigauri.org>                            #
#           Dr. Alexander K. Seewald <alex@seewald.at>                          #
#                             Autofocus algorithm                               #
#           Flemming Frandsen <dren.dk@gmail.com>                               #
#           George Sedov <radist.morse@gmail.com>                               #
#                                                                               #
# This program is free software; you can redistribute it and/or modify          #
# it under the terms of the GNU General Public License as published by          #
# the Free Software Foundation; either version 2 of the License, or             #
# (at your option) any later version.                                           #
#                                                                               #
# This program is distributed in the hope that it will be useful,               #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                 #
# GNU General Public License for more details.                                  #
#                                                                               #
# You should have received a copy of the GNU General Public License             #
# along with this program; if not, write to the Free Software                   #
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA     #
#                                                                               #
********************************************************************************/


#include <SDL/SDL.h>
#include <glib.h>
#include <glib/gprintf.h>
/* support for internationalization - i18n */
#include <glib/gi18n.h>
/* locale.h is needed if -O0 used (no optimiztions)  */
/* otherwise included from libintl.h on glib/gi18n.h */
#include <locale.h>
#include <signal.h>
#include <fcntl.h>		/* for fcntl, O_NONBLOCK */
#include <gtk/gtk.h>
#include <portaudio.h>

#include "config.h"
#include "globals.h"
#include "guvcview.h"

#define VDIN_DYNCTRL_OK            3
#define __VMUTEX &videoIn->common.mutex

static Uint32 SDL_VIDEO_Flags =
        SDL_ANYFORMAT | SDL_RESIZABLE;

static const SDL_VideoInfo *info;

static void
shutd (gint restart, struct ALL_DATA *all_data)
{
	gchar videodevice[16];
	struct GWIDGET *gwidget = all_data->gwidget;
	//gchar *EXEC_CALL = all_data->EXEC_CALL;

	//struct paRecordData *pdata = all_data->pdata;
	struct GLOBAL *global = all_data->global;
	VDIN_T *videoIn = all_data->videoIn;

	gboolean control_only = (global->control_only || global->add_ctrls);
	gboolean no_display = global->no_display;
	GMainLoop *main_loop = gwidget->main_loop;

	/* wait for the video thread */
	if(!(control_only))
	{
		if (global->debug) g_print("Shuting Down Thread\n");
		__LOCK_MUTEX(__VMUTEX);
            global->signalquit = TRUE;
			//videoIn->common.signalquit=TRUE;
		__UNLOCK_MUTEX(__VMUTEX);
		__THREAD_JOIN(all_data->video_thread);
		if (global->debug) g_print("Video Thread finished\n");
	}

	/* destroys fps timer*/
	if (global->timer_id > 0) g_source_remove(global->timer_id);
	/* destroys udev device event check timer*/
	if (global->udev_timer_id > 0) g_source_remove(global->udev_timer_id);

	if(!no_display)
	{
	    gtk_window_get_size(GTK_WINDOW(gwidget->mainwin),&(global->winwidth),&(global->winheight));//mainwin or widget
	}

	g_snprintf(videodevice, 15, "%s", global->videodevice);

	gwidget = NULL;
	//pdata = NULL;
	global = NULL;
	videoIn = NULL;

	//end gtk or glib main loop
	if(!no_display)
		gtk_main_quit();
	else
		g_main_loop_quit(main_loop);

}
static int shutd_timer(gpointer data)
{
    /*stop video capture*/
    shutd (0, data);
    
    return (FALSE);/*destroys the timer*/
}

static SDL_Overlay * video_init(void *data, SDL_Surface **pscreen)
{
    struct ALL_DATA *all_data = (struct ALL_DATA *) data;
    struct GLOBAL *global = all_data->global;

    int width = global->width;
    int height = global->height;

    if (*pscreen == NULL) //init SDL
    {
        char driver[128];
        /*----------------------------- Test SDL capabilities ---------------------*/
        if (SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER) < 0)
        {
            g_printerr("Couldn't initialize SDL: %s\n", SDL_GetError());
            exit(1);
        }

        /* For this version, we will use hardware acceleration as default*/
        if(global->hwaccel)
        {
            if ( ! getenv("SDL_VIDEO_YUV_HWACCEL") ) putenv("SDL_VIDEO_YUV_HWACCEL=1");
            if ( ! getenv("SDL_VIDEO_YUV_DIRECT") ) putenv("SDL_VIDEO_YUV_DIRECT=1");
        }
        else
        {
            if ( ! getenv("SDL_VIDEO_YUV_HWACCEL") ) putenv("SDL_VIDEO_YUV_HWACCEL=0");
            if ( ! getenv("SDL_VIDEO_YUV_DIRECT") ) putenv("SDL_VIDEO_YUV_DIRECT=0");
        }

        if (SDL_VideoDriverName(driver, sizeof(driver)) && global->debug)
        {
            g_print("Video driver: %s\n", driver);
        }

        info = SDL_GetVideoInfo();

        if (info->wm_available && global->debug) g_print("A window manager is available\n");

        if (info->hw_available)
        {
            if (global->debug)
                g_print("Hardware surfaces are available (%dK video memory)\n", info->video_mem);

            SDL_VIDEO_Flags |= SDL_HWSURFACE;
            SDL_VIDEO_Flags |= SDL_DOUBLEBUF;
        }
        else
        {
            SDL_VIDEO_Flags |= SDL_SWSURFACE;
        }

        if (info->blit_hw)
        {
            if (global->debug) g_print("Copy blits between hardware surfaces are accelerated\n");

            SDL_VIDEO_Flags |= SDL_ASYNCBLIT;
        }

        if(!global->desktop_w) global->desktop_w = info->current_w; //get desktop width
        if(!global->desktop_h) global->desktop_h = info->current_h; //get desktop height

        if (global->debug)
        {
            if (info->blit_hw_CC) g_print ("Colorkey blits between hardware surfaces are accelerated\n");
            if (info->blit_hw_A) g_print("Alpha blits between hardware surfaces are accelerated\n");
            if (info->blit_sw) g_print ("Copy blits from software surfaces to hardware surfaces are accelerated\n");
            if (info->blit_sw_CC) g_print ("Colorkey blits from software surfaces to hardware surfaces are accelerated\n");
            if (info->blit_sw_A) g_print("Alpha blits from software surfaces to hardware surfaces are accelerated\n");
            if (info->blit_fill) g_print("Color fills on hardware surfaces are accelerated\n");
        }

        SDL_WM_SetCaption(global->WVcaption, NULL);

        /* enable key repeat */
        SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY,SDL_DEFAULT_REPEAT_INTERVAL);
    }
    /*------------------------------ SDL init video ---------------------*/

    if(global->debug)
        g_print("(Desktop resolution = %ix%i)\n", global->desktop_w, global->desktop_h);
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
        if ((width > global->desktop_w) || (height > global->desktop_h))
        {
            width = global->desktop_w; /*use desktop video resolution*/
            height = global->desktop_h;
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
        if ((bpp != 32) && global->debug) g_print("recomended color depth = %i\n", bpp);
        global->bpp = bpp;
    }

    *pscreen = SDL_SetVideoMode(
        width,
        height,
        global->bpp,
        SDL_VIDEO_Flags);

    if(*pscreen == NULL)
    {
        return (NULL);
    }
    //use requested resolution for overlay even if not available as video mode
    SDL_Overlay* overlay=NULL;
    overlay = SDL_CreateYUVOverlay(global->width, global->height,
        SDL_YUY2_OVERLAY, *pscreen);

    SDL_ShowCursor(SDL_DISABLE);
    return (overlay);
}

/*-------------------------------- Main Video Loop ---------------------------*/
/* run in a thread (SDL overlay)*/
static void *main_loop(void *data)
{
    struct ALL_DATA *all_data = (struct ALL_DATA *) data;

    struct GLOBAL *global = all_data->global;
    VDIN_T *videoIn = all_data->videoIn;

    struct particle* particles = NULL; //for the particles video effect

    SDL_Event event;
    /*the main SDL surface*/
    SDL_Surface *pscreen = NULL;
    SDL_Overlay *overlay = NULL;
    SDL_Rect drect;

    SAMPLE vuPeak[2];  // The maximum vuLevel seen recently
    int vuPeakFreeze[2]; // The vuPeak values will be frozen for this many frames.
    vuPeak[0] = vuPeak[1] = 0;
    vuPeakFreeze[0] = vuPeakFreeze[1] = 0;

    BYTE *p = NULL;

    global->signalquit = FALSE;

    /*------------------------------ SDL init video ---------------------*/
    if(!global->no_display)
    {
        overlay = video_init(data, &(pscreen));

        if(overlay == NULL)
        {
            g_print("FATAL: Couldn't create yuv overlay - please disable hardware accelaration\n");
            global->signalquit = TRUE; /*exit video thread*/
        }
        else
        {
            p = (unsigned char *) overlay->pixels[0];

            drect.x = 0;
            drect.y = 0;
            drect.w = pscreen->w;
            drect.h = pscreen->h;
        }
    }

    while (!global->signalquit)
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
                pscreen =
                    SDL_SetVideoMode(event.resize.w,
                            event.resize.h,
                            global->bpp,
                            SDL_VIDEO_Flags);
                drect.w = event.resize.w;
                drect.h = event.resize.h;
            }
            if(event.type==SDL_QUIT)
            {
                //shutDown
                g_timeout_add(200, shutd_timer, all_data);
            }
        }

    }/*loop end*/


    if (global->debug) g_print("Thread terminated...\n");
    p = NULL;
    if(particles) g_free(particles);
    particles=NULL;

    if (global->debug) g_print("cleaning Thread allocations: 100%%\n");
    fflush(NULL);//flush all output buffers

    if(!global->no_display)
    {
        if(overlay)
            SDL_FreeYUVOverlay(overlay);
        //SDL_FreeSurface(pscreen);

        SDL_Quit();
    }

    if (global->debug) g_print("Video thread completed\n");

    global = NULL;
    videoIn = NULL;
    return ((void *) 0);
}


/*----------------------------- globals --------------------------------------*/

struct paRecordData *pdata = NULL;
struct GLOBAL *global = NULL;
struct focusData *AFdata = NULL;
VDIN_T *videoIn = NULL;
struct VideoFormatData *videoF = NULL;

/*controls data*/
struct VidState *s = NULL;
/*global widgets*/
struct GWIDGET *gwidget = NULL;

/*thread definitions*/
//__THREAD_TYPE video_thread;

/*
 * Unix signals that are cought are written to a pipe. The pipe connects
 * the unix signal handler with GTK's event loop. The array signal_pipe will
 * hold the file descriptors for the two ends of the pipe (index 0 for
 * reading, 1 for writing).
 */
int signal_pipe[2];

/*
 * The unix signal handler.
 * Write any unix signal into the pipe. The writing end of the pipe is in
 * non-blocking mode. If it is full (which can only happen when the
 * event loop stops working) signals will be dropped.
 */
void pipe_signals(int signal)
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
gboolean deliver_signal(GIOChannel *source, GIOCondition cond, gpointer data)
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
     		shutd(0, (struct ALL_DATA *)data);//shutDown
            //shutd (gint restart, struct ALL_DATA *all_data)
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

	/*print package name and version*/
	g_print("%s\n", PACKAGE_STRING);

	/*structure containing all shared data - passed in callbacks*/
	struct ALL_DATA all_data;
	memset(&all_data,0,sizeof(struct ALL_DATA));

	/*allocate global variables*/
	global = g_new0(struct GLOBAL, 1);
	initGlobals(global);

	/*------------------------- reads configuration file ---------------------*/

	//sets local control_only flag - prevents several initializations/allocations
	gboolean control_only = (global->control_only || global->add_ctrls) ;

    if(global->no_display && global->control_only )
    {
	g_printerr("incompatible options (control_only and no_display): enabling display");
	global->no_display = FALSE;
    }

	/*---------------------------------- Allocations -------------------------*/

	gwidget = g_new0(struct GWIDGET, 1);
	gwidget->vid_widget_state = TRUE;

	/* widgets */
	GtkWidget *scroll1;
	GtkWidget *Tab1;
	GtkWidget *Tab1Label;
	GtkWidget *Tab1Icon;
	GtkWidget *ImgButton_Img;
	GtkWidget *VidButton_Img;
	GtkWidget *QButton_Img;
	GtkWidget *HButtonBox;


	if(!global->no_display)
	{
		if(!gtk_init_check(&argc, &argv))
		{
			g_printerr("GUVCVIEW: can't open display: changing to no_display mode\n");
			global->no_display = TRUE; /*if we can't open the display fallback to no_display mode*/
		}
	}

    if(!global->no_display)
    {
		g_set_application_name(_("LEOPARD Video Capture"));
		g_setenv("PULSE_PROP_media.role", "video", TRUE); //needed for Pulse Audio

        /* make sure the type is realized so that we can change the properties*/
        g_type_class_unref (g_type_class_ref (GTK_TYPE_BUTTON));
        /* make sure gtk-button-images property is set to true (defaults to false in karmic)*/
        g_object_set (gtk_settings_get_default (), "gtk-button-images", TRUE, NULL);

        //get screen resolution
        if((!global->desktop_w) || (!global->desktop_h))
        {
            GdkScreen* screen = NULL;
            global->desktop_w = gdk_screen_get_width(screen);
            global->desktop_h = gdk_screen_get_height(screen);
        }
        if(global->debug)
            g_print("Screen resolution is (%d x %d)\n", global->desktop_w, global->desktop_h);

        if((global->winwidth > global->desktop_w) && (global->desktop_w > 0))
            global->winwidth = global->desktop_w;
        if((global->winheight > global->desktop_h) && (global->desktop_h > 0))
            global->winheight = global->desktop_h;

    }

	/*----------------------- init videoIn structure --------------------------*/
	videoIn = g_new0(VDIN_T, 1);

    /*set structure with all global allocations*/
    all_data.pdata = pdata;
    all_data.global = global;
    all_data.AFdata = AFdata; /*not allocated yet*/
    all_data.videoIn = videoIn;
    all_data.videoF = videoF;
    all_data.gwidget = gwidget;
    all_data.h264_controls = NULL; /*filled by add_uvc_h264_controls_tab */
    all_data.s = s;

    global->width =  WIDTH;
    global->height = HEIGHT;

    VD_INIT("/dev/video0", videoIn);

    if(!(global->no_display))
    {
        gwidget->maintable = gtk_box_new(GTK_ORIENTATION_VERTICAL, 2);

        gtk_widget_show (gwidget->maintable);

        gwidget->boxh = gtk_notebook_new();

        gtk_widget_show (gwidget->boxh);

        scroll1=gtk_scrolled_window_new(NULL,NULL);
        gtk_scrolled_window_set_placement(GTK_SCROLLED_WINDOW(scroll1), GTK_CORNER_TOP_LEFT);

        //viewport is only needed for gtk < 3.8
        //for 3.8 and above s->table can be directly added to scroll1
        GtkWidget* viewport = gtk_viewport_new(NULL,NULL);
        gtk_widget_show(viewport);

        gtk_container_add(GTK_CONTAINER(scroll1), viewport);
        gtk_widget_show(scroll1);

        Tab1 = gtk_grid_new();
        Tab1Label = gtk_label_new(_("Image Controls"));
        gtk_widget_show (Tab1Label);
        /** check for files */
        gchar* Tab1IconPath = g_strconcat (PACKAGE_DATA_DIR,"/pixmaps/guvcview/image_controls.png",NULL);
        /** don't test for file - use default empty image if load fails */
        /** get icon image*/
        Tab1Icon = gtk_image_new_from_file(Tab1IconPath);
        g_free(Tab1IconPath);
        gtk_widget_show (Tab1Icon);
        gtk_grid_attach (GTK_GRID(Tab1), Tab1Icon, 0, 0, 1, 1);
        gtk_grid_attach (GTK_GRID(Tab1), Tab1Label, 1, 0, 1, 1);

        gtk_widget_show (Tab1);

        gtk_notebook_append_page(GTK_NOTEBOOK(gwidget->boxh),scroll1,Tab1);

        /*---------------------- Add  Buttons ---------------------------------*/
        HButtonBox = gtk_button_box_new(GTK_ORIENTATION_HORIZONTAL);
        gtk_widget_set_halign (HButtonBox, GTK_ALIGN_FILL);
		gtk_widget_set_hexpand (HButtonBox, TRUE);
        gtk_button_box_set_layout(GTK_BUTTON_BOX(HButtonBox),GTK_BUTTONBOX_SPREAD);
        gtk_box_set_homogeneous(GTK_BOX(HButtonBox),TRUE);

		gtk_widget_show(HButtonBox);

		/** Attach the buttons */
		gtk_box_pack_start(GTK_BOX(gwidget->maintable), HButtonBox, FALSE, TRUE, 2);
		/** Attach the notebook (tabs) */
		gtk_box_pack_start(GTK_BOX(gwidget->maintable), gwidget->boxh, TRUE, TRUE, 2);

        //gwidget->quitButton=gtk_button_new_from_stock(GTK_STOCK_QUIT);

        gchar* icon1path = g_strconcat (PACKAGE_DATA_DIR,"/pixmaps/guvcview/guvcview.png",NULL);
        g_free(icon1path);

        if(!control_only)/*control_only exclusion Image and video buttons*/
        {
            if(global->image_timer)
            {	/*image auto capture*/
                gwidget->CapImageButt=gtk_button_new_with_label (_("Stop Auto (I)"));
            }
            else
            {
                gwidget->CapImageButt=gtk_button_new_with_label (_("Cap. Image (I)"));
            }

            if (global->Capture_time > 0)
            {	/*vid capture enabled from start*/
                gwidget->CapVidButt=gtk_toggle_button_new_with_label (_("Stop Video (V)"));
                gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gwidget->CapVidButt), TRUE);
            }
            else
            {
                gwidget->CapVidButt=gtk_toggle_button_new_with_label (_("Cap. Video (V)"));
                gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gwidget->CapVidButt), FALSE);
            }

            /*add images to Buttons and top window*/
            /*check for files*/

            gchar* pix1path = g_strconcat (PACKAGE_DATA_DIR,"/pixmaps/guvcview/movie.png",NULL);
            if (g_file_test(pix1path,G_FILE_TEST_EXISTS))
            {
                VidButton_Img = gtk_image_new_from_file (pix1path);

                gtk_button_set_image(GTK_BUTTON(gwidget->CapVidButt),VidButton_Img);
                gtk_button_set_image_position(GTK_BUTTON(gwidget->CapVidButt),GTK_POS_TOP);
                //gtk_widget_show (gwidget->VidButton_Img);
            }
            //else g_print("couldn't load %s\n", pix1path);
            gchar* pix2path = g_strconcat (PACKAGE_DATA_DIR,"/pixmaps/guvcview/camera.png",NULL);
            if (g_file_test(pix2path,G_FILE_TEST_EXISTS))
            {
                ImgButton_Img = gtk_image_new_from_file (pix2path);

                gtk_button_set_image(GTK_BUTTON(gwidget->CapImageButt),ImgButton_Img);
                gtk_button_set_image_position(GTK_BUTTON(gwidget->CapImageButt),GTK_POS_TOP);
                //gtk_widget_show (ImgButton_Img);
            }
            g_free(pix1path);
            g_free(pix2path);
            gtk_box_pack_start(GTK_BOX(HButtonBox),gwidget->CapImageButt,TRUE,TRUE,2);
            gtk_box_pack_start(GTK_BOX(HButtonBox),gwidget->CapVidButt,TRUE,TRUE,2);
            gtk_toggle_button_set_mode (GTK_TOGGLE_BUTTON (gwidget->CapVidButt), FALSE);
            gtk_widget_show (gwidget->CapImageButt);
            gtk_widget_show (gwidget->CapVidButt);

        }/*end of control_only exclusion*/

        gchar* pix3path = g_strconcat (PACKAGE_DATA_DIR,"/pixmaps/guvcview/close.png",NULL);
        if (g_file_test(pix3path,G_FILE_TEST_EXISTS))
        {
            QButton_Img = gtk_image_new_from_file (pix3path);

            gtk_button_set_image(GTK_BUTTON(gwidget->quitButton),QButton_Img);
            gtk_button_set_image_position(GTK_BUTTON(gwidget->quitButton),GTK_POS_TOP);
        }

        /*must free path strings*/
        g_free(pix3path);

        gtk_box_pack_start(GTK_BOX(HButtonBox), gwidget->quitButton,TRUE,TRUE,2);

        gtk_widget_show_all (gwidget->quitButton);


		gwidget->status_bar = gtk_statusbar_new();
		gwidget->status_warning_id = gtk_statusbar_get_context_id (GTK_STATUSBAR(gwidget->status_bar), "warning");

        gtk_widget_show(gwidget->status_bar);

        gtk_box_pack_start(GTK_BOX(gwidget->maintable), gwidget->status_bar, FALSE, FALSE, 2);
    }


	if (!control_only) /*control_only exclusion*/
	{
		/*------------------ Creating the video thread ---------------*/
		if( __THREAD_CREATE(&all_data.video_thread, main_loop, (void *) &all_data))
		{
			g_printerr("Video thread creation failed\n");

		}

	}/*end of control_only exclusion*/

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
  	g_io_add_watch(g_signal_in, G_IO_IN | G_IO_PRI, deliver_signal, &all_data);


	/* The last thing to get called (gtk or glib main loop)*/
	if(global->debug)
		g_print("Starting main loop \n");

	if(!global->no_display)
		gtk_main();
	else
	{
		gwidget->main_loop = g_main_loop_new(NULL, TRUE);
		g_main_loop_run(gwidget->main_loop);
	}

	//free all_data allocations
	free(all_data.gwidget);
	if(all_data.h264_controls != NULL)
		free(all_data.h264_controls);

	g_print("Closing GTK... OK\n");
	return 0;
}

 
