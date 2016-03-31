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
#define __COND_TYPE pthread_cond_t
#define __INIT_MUTEX(m) ( pthread_mutex_init(m, NULL) )

#define __INIT_COND(c)  ( pthread_cond_init (c, NULL) )

typedef int8_t   INT8;
typedef uint8_t  UINT8;
typedef int16_t  INT16;
typedef uint16_t UINT16;
typedef int32_t  INT32;
typedef uint32_t UINT32;
typedef int64_t  INT64;
typedef uint64_t UINT64;
typedef uint64_t QWORD;
typedef uint32_t DWORD;
typedef uint16_t WORD;
typedef uint8_t  BYTE;
typedef unsigned int LONG;
typedef unsigned int UINT;
typedef unsigned long long ULLONG;
typedef unsigned long      ULONG;

typedef char* pchar;

#define DEBUG (0)

#define INCPANTILT 64 // 1°

#define WINSIZEX 560
#define WINSIZEY 560

#define AUTO_EXP 8
#define MAN_EXP	1

#define DHT_SIZE 432

#define DEFAULT_WIDTH 640
#define DEFAULT_HEIGHT 480

#define DEFAULT_FPS	25
#define DEFAULT_FPS_NUM 1
#define SDL_WAIT_TIME 30 /*SDL - Thread loop sleep time */

/*clip value between 0 and 255*/
#define CLIP(value) (BYTE)(((value)>0xFF)?0xff:(((value)<0)?0:(value)))

/*MAX macro - gets the bigger value*/
#ifndef MAX
#define MAX(a,b) (((a) < (b)) ? (b) : (a))
#endif


struct GLOBAL
{

	__MUTEX_TYPE mutex;    //global struct mutex
	__MUTEX_TYPE file_mutex; //video file mutex
	__COND_TYPE  IO_cond;      //IO thread semaphore

	//VidBuff *videoBuff;    //video Buffer
	int video_buff_size;   //size in frames of video buffer

	char *videodevice;     // video device (def. /dev/video0)
	char *confPath;        //configuration file path
	char *vidfile;         //video filename passed through argument options with -n
	char *WVcaption;       //video preview title bar caption
	char *mode;            //mjpg (default)
	pchar* vidFPath;       //video path [0] - filename  [1] - dir
	pchar* imgFPath;       //image path [0] - filename  [1] - dir
	pchar* profile_FPath;  //profile path [0] - filename  [1] - dir

	BYTE *jpeg;            // jpeg buffer

	int64_t av_drift;      // amount of A/V time correction
	UINT64 Vidstarttime;   //video start time
	UINT64 Vidstoptime;    //video stop time
	QWORD v_ts;            //video time stamp
	QWORD a_ts;            //audio time stamp
	uint64_t vid_inc;      //video name increment
	uint64_t framecount;   //video frame count
	DWORD frmCount;        //frame count for fps display calc
	uint64_t image_inc;    //image name increment

	int vid_sleep;         //video thread sleep time (0 by default)
	int Capture_time;      //video capture time passed through argument options with -t
	int imgFormat;         //image format: 0-"jpg", 1-"png", 2-"bmp"
	int VidCodec;          //0-"MJPG"  1-"YUY2" 2-"DIB "(rgb32) 3-....
	int VidCodec_ID;       //lavc codec ID
	int AudCodec;          //0-PCM 1-MPG2 3-...
	int VidFormat;         //0-AVI 1-MKV ....
	int Sound_API;         //audio API: 0-PORTAUDIO 1-PULSEAUDIO
	int Sound_SampRate;    //audio sample rate
	int Sound_SampRateInd; //audio sample rate combo index
	int Sound_numInputDev; //number of audio input devices
	int Sound_DefDev;      //audio default device index
	int Sound_UseDev;      //audio used device index
	int Sound_NumChan;     //audio number of channels
	int Sound_NumChanInd;  //audio number of channels combo index
	WORD Sound_Format;     //audio codec - fourcc (avilib.h)
	uint64_t Sound_delay;  //added sound delay (in nanosec)
	int PanStep;           //step angle for Pan
	int TiltStep;          //step angle for Tilt
	int FpsCount;          //frames counter for fps calc
	int timer_id;          //fps count timer
	int image_timer_id;    //auto image capture timer
    int udev_timer_id;     //timer id for udev device events check
	int disk_timer_id;     //timer id for disk check (free space)
	int image_timer;       //auto image capture time
	int image_npics;       //number of captures
	int image_picn;        //capture number
	int bpp;               //current bytes per pixel
	int hwaccel;           //use hardware acceleration
	int desktop_w;         //Desktop width
	int desktop_h;         //Desktop height
	int width;             //frame width
	int height;            //frame height
	int winwidth;          //control windoe width
	int winheight;         //control window height
	int Frame_Flags;       //frame filter flags
	int osdFlags;          // Flags to control onscreen display
	int skip_n;            //initial frames to skip
	int w_ind;             //write frame index
	int r_ind;             //read  frame index
	int default_action;    // 0 for taking picture, 1 for video
	int lctl_method;       // 0 for control id loop, 1 for next_ctrl flag method
	int uvc_h264_unit;     //uvc h264 unit id, if <= 0 then uvc h264 is not supported

	float DispFps;         //fps value

    gboolean no_display;   //flag if guvcview will present the gui or not.
	gboolean exit_on_close;//exit guvcview after closing video when capturing from start
	gboolean Sound_enable; //Enable/disable Sound (Def. enable)
	gboolean AFcontrol;    //Autofocus control flag (exists or not)
	gboolean autofocus;    //autofocus flag (enable/disable)
	gboolean flg_config;   //flag confPath if set in args
	gboolean lprofile;     //flag for command line -l option
	gboolean flg_npics;    //flag npics if set in args
	gboolean flg_hwaccel;  //flag hwaccel if set in args
	gboolean flg_res;      //flag resol if set in args
	gboolean flg_mode;     //flag mode if set in args
	gboolean flg_imgFPath; //flag imgFPath if set in args
	gboolean flg_FpsCount; //flag FpsCount if set in args
	gboolean debug;        //debug mode flag (--verbose)
	gboolean VidButtPress;
	gboolean control_only; //if set don't stream video (enables image control in other apps e.g. ekiga, skype, mplayer)
	gboolean change_res;   //flag for reseting resolution
	gboolean add_ctrls;    //flag for exiting after adding extension controls
	gboolean monotonic_pts;//flag if we are using monotonic or real pts
    gboolean signalquit;
};


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

#define __AMUTEX &pdata->mutex
#define __GMUTEX &global->mutex
#define __FMUTEX &global->file_mutex
#define __GCOND  &global->IO_cond

#define LIST_CTL_METHOD_NEXT_FLAG 1

static int initGlobals (struct GLOBAL *global)
{
	__INIT_MUTEX( __GMUTEX );
	__INIT_MUTEX( __FMUTEX );
	__INIT_COND( __GCOND );   /* Initialized video buffer semaphore */

	global->debug = DEBUG;

	const gchar *home = g_get_home_dir();

	global->videodevice = g_strdup("/dev/video0");

	global->confPath = g_strjoin("/", home, ".config", "guvcview", NULL);
	int ret = g_mkdir_with_parents(global->confPath, 0777);
	if(ret)
		fprintf(stderr, "Couldn't create configuration dir: %s \n", global->confPath);

	g_free(global->confPath);
	global->confPath = g_strjoin("/", home, ".config", "guvcview", "video0", NULL);

	global->vidFPath = g_new(pchar, 2);

	global->imgFPath = g_new(pchar, 2);

	global->profile_FPath = g_new(pchar, 2);

	global->vidFPath[1] = g_strdup(home);

	global->imgFPath[1] = g_strdup(home);

	global->profile_FPath[1] = g_strdup(home);

	global->vidFPath[0] = g_strdup("guvcview_video.mkv");

	global->imgFPath[0] = g_strdup("guvcview_image.jpg");

	global->profile_FPath[0] = g_strdup("default.gpfl");

	global->WVcaption = g_new(char, 32);

	g_sprintf(global->WVcaption,"LI-USB30-M021");

	global->image_inc = 1; //increment filename by default
	global->vid_inc = 1;   //increment filename by default

	global->vid_sleep=0;
	global->vidfile=NULL; /*vid filename passed through argument options with -n */
	global->Capture_time=0; /*vid capture time passed through argument options with -t */
	global->lprofile=0; /* flag for -l command line option*/


	global->av_drift=0;
	global->Vidstarttime=0;
	global->Vidstoptime=0;
	global->framecount=0;
	global->w_ind=0;
	global->r_ind=0;

	global->FpsCount=0;

	global->disk_timer_id=0;
	global->timer_id=0;
	global->image_timer_id=0;
	global->image_timer=0;
	global->image_npics=9999;/*default max number of captures*/
	global->image_picn =0;
	global->frmCount=0;
	global->PanStep=2;/*2 degree step for Pan*/
	global->TiltStep=2;/*2 degree step for Tilt*/
	global->DispFps=0;
	global->bpp = 0; //current bytes per pixel
	global->hwaccel = 1; //use hardware acceleration
	global->desktop_w = 0;
	global->desktop_h = 0;
	global->width = DEFAULT_WIDTH;
	global->height = DEFAULT_HEIGHT;
	global->winwidth=WINSIZEX;
	global->winheight=WINSIZEY;

	global->default_action=0;

	global->mode = g_new(char, 6);
	g_snprintf(global->mode, 5, "mjpg");

	global->osdFlags = 0;

    global->no_display = FALSE;
	global->exit_on_close = FALSE;
	global->skip_n=0;
	global->jpeg=NULL;
	global->uvc_h264_unit = 0; //not supported by default

	/* reset with videoIn parameters */
	global->autofocus = FALSE;
	global->AFcontrol = FALSE;
	global->VidButtPress = FALSE;
	global->change_res = FALSE;
	global->add_ctrls = FALSE;
	global->lctl_method = LIST_CTL_METHOD_NEXT_FLAG; //next_ctrl flag method
	return (0);
}

/* Must set this as global so they */
/* can be set from any callback.   */

struct GWIDGET
{
	/*the main loop : only needed for no_display option*/
	GMainLoop *main_loop;

	/* The main window*/
	GtkWidget *mainwin;
	/* A restart Dialog */
	GtkWidget *restartdialog;
	/*Paned containers*/
	GtkWidget *maintable;
	GtkWidget *boxh;

	//group list for menu video codecs
	GSList *vgroup;
	//group list for menu audio codecs
	GSList *agroup;

	//menu top widgets
	GtkWidget *menu_photo_top;
	GtkWidget *menu_video_top;

	GtkWidget *status_bar;

	GtkWidget *label_SndAPI;
	GtkWidget *SndAPI;
	GtkWidget *SndEnable;
	GtkWidget *SndSampleRate;
	GtkWidget *SndDevice;
	GtkWidget *SndNumChan;
	GtkWidget *SndComp;
	/*must be called from main loop if capture timer enabled*/
	GtkWidget *ImageType;
	GtkWidget *CapImageButt;
	GtkWidget *CapVidButt;
	GtkWidget *Resolution;
	GtkWidget *InpType;
	GtkWidget *FrameRate;
	GtkWidget *Devices;
	GtkWidget *jpeg_comp;
	GtkWidget *quitButton;

	gboolean vid_widget_state;
	int status_warning_id;
};

/* uvc H264 control widgets */
struct uvc_h264_gtkcontrols
{
	GtkWidget* FrameInterval;
	GtkWidget* BitRate;
	GtkWidget* Hints_res;
	GtkWidget* Hints_prof;
	GtkWidget* Hints_ratecontrol;
	GtkWidget* Hints_usage;
	GtkWidget* Hints_slicemode;
	GtkWidget* Hints_sliceunit;
	GtkWidget* Hints_view;
	GtkWidget* Hints_temporal;
	GtkWidget* Hints_snr;
	GtkWidget* Hints_spatial;
	GtkWidget* Hints_spatiallayer;
	GtkWidget* Hints_frameinterval;
	GtkWidget* Hints_leakybucket;
	GtkWidget* Hints_bitrate;
	GtkWidget* Hints_cabac;
	GtkWidget* Hints_iframe;
	GtkWidget* Resolution;
	GtkWidget* SliceUnits;
	GtkWidget* SliceMode;
	GtkWidget* Profile;
	GtkWidget* Profile_flags;
	GtkWidget* IFramePeriod;
	GtkWidget* EstimatedVideoDelay;
	GtkWidget* EstimatedMaxConfigDelay;
	GtkWidget* UsageType;
	//GtkWidget* UCConfig;
	GtkWidget* RateControlMode;
	GtkWidget* RateControlMode_cbr_flag;
	GtkWidget* TemporalScaleMode;
	GtkWidget* SpatialScaleMode;
	GtkWidget* SNRScaleMode;
	GtkWidget* StreamMuxOption;
	GtkWidget* StreamMuxOption_aux;
	GtkWidget* StreamMuxOption_mjpgcontainer;
	GtkWidget* StreamFormat;
	GtkWidget* EntropyCABAC;
	GtkWidget* Timestamp;
	GtkWidget* NumOfReorderFrames;
	GtkWidget* PreviewFlipped;
	GtkWidget* View;
	GtkWidget* StreamID;
	GtkWidget* SpatialLayerRatio;
	GtkWidget* LeakyBucketSize;
	GtkWidget* probe_button;
	GtkWidget* commit_button;
};

struct ALL_DATA
{
	struct paRecordData *pdata;
	struct GLOBAL *global;
	struct focusData *AFdata;
	VDIN_T *videoIn;
	struct VideoFormatData *videoF;
	struct GWIDGET *gwidget;
	struct uvc_h264_gtkcontrols  *h264_controls;
	struct VidState *s;
	__THREAD_TYPE video_thread;
	__THREAD_TYPE audio_thread;
	__THREAD_TYPE IO_thread;
};


#define VDIN_DYNCTRL_OK            3

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

	gboolean control_only = (global->control_only || global->add_ctrls);
	gboolean no_display = global->no_display;
	GMainLoop *main_loop = gwidget->main_loop;

	/* wait for the video thread */
	if(!(control_only))
	{
		if (global->debug) g_print("Shuting Down Thread\n");
        global->signalquit = TRUE;
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

 
