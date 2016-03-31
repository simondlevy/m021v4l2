#ifndef GLOBALS_H
#define GLOBALS_H

#include <inttypes.h>
#include <sys/types.h>
#include <glib.h>
#include <pthread.h>

#define CLEAR_LINE "\x1B[K"

#ifdef WORDS_BIGENDIAN
  #define BIGENDIAN 1
#else
  #define BIGENDIAN 0
#endif

#define IO_MMAP 1
#define IO_READ 2

#define ODD(x) ((x%2)?TRUE:FALSE)

#define __THREAD_TYPE pthread_t
#define __THREAD_CREATE(t,f,d) (pthread_create(t,NULL,f,d))
#define __THREAD_JOIN(t) (pthread_join(t, NULL))


#define __MUTEX_TYPE pthread_mutex_t
#define __COND_TYPE pthread_cond_t
#define __INIT_MUTEX(m) ( pthread_mutex_init(m, NULL) )
#define __CLOSE_MUTEX(m) ( pthread_mutex_destroy(m) )
#define __LOCK_MUTEX(m) ( pthread_mutex_lock(m) )
#define __UNLOCK_MUTEX(m) ( pthread_mutex_unlock(m) )

#define __INIT_COND(c)  ( pthread_cond_init (c, NULL) )
#define __CLOSE_COND(c) ( pthread_cond_destroy(c) )
#define __COND_BCAST(c) ( pthread_cond_broadcast(c) )
#define __COND_TIMED_WAIT(c,m,t) ( pthread_cond_timedwait(c,m,t) )

/*next index of ring buffer with size elements*/
#define NEXT_IND(ind,size) ind++;if(ind>=size) ind=0
/*previous index of ring buffer with size elements*/
//#define PREV_IND(ind,size) ind--;if(ind<0) ind=size-1

#define VIDBUFF_SIZE 45    //number of video frames in the ring buffer

#define MPG_NUM_SAMP 1152  //number of samples in a audio MPEG frame
#define AUDBUFF_SIZE 2     //number of audio mpeg frames in each audio buffer
                           // direct impact on latency as buffer is only processed when full
#define AUDBUFF_NUM  80    //number of audio buffers
//#define MPG_NUM_FRAMES 2   //number of MPEG frames in a audio frame

typedef uint64_t QWORD;
typedef uint32_t DWORD;
typedef uint16_t WORD;
typedef uint8_t  BYTE;
typedef unsigned int LONG;
typedef unsigned int UINT;

typedef unsigned long long ULLONG;
typedef unsigned long      ULONG;

typedef char* pchar;

typedef int8_t     INT8;
typedef uint8_t    UINT8;
typedef int16_t    INT16;
typedef uint16_t   UINT16;
typedef int32_t    INT32;
typedef uint32_t   UINT32;
typedef int64_t    INT64;
typedef uint64_t   UINT64;

typedef float SAMPLE;

/* 0 is device default*/
static const int stdSampleRates[] =
{
	0, 8000,  9600, 11025, 12000,
	16000, 22050, 24000,
	32000, 44100, 48000,
	88200, 96000,
	-1   /* Negative terminated list. */
};

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
	int cap_meth;          //capture method: 1-mmap 2-read
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
	gboolean flg_cap_meth; //flag if cap_meth is set in args
	gboolean debug;        //debug mode flag (--verbose)
	gboolean VidButtPress;
	gboolean control_only; //if set don't stream video (enables image control in other apps e.g. ekiga, skype, mplayer)
	gboolean change_res;   //flag for reseting resolution
	gboolean add_ctrls;    //flag for exiting after adding extension controls
	gboolean monotonic_pts;//flag if we are using monotonic or real pts
    gboolean signalquit;
};

/*----------------------------- prototypes ------------------------------------*/
int initGlobals(struct GLOBAL *global);

int closeGlobals(struct GLOBAL *global);


#endif

