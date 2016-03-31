/*******************************************************************************#
#           guvcview              http://guvcview.sourceforge.net               #
#                                                                               #
#           Paulo Assis <pj.assis@gmail.com>                                    #
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

#ifndef GLOBALS_H
#define GLOBALS_H

#include <glib.h>
#include <pthread.h>
#include "defs.h"

typedef struct _VidBuff
{
	gboolean used;
	QWORD time_stamp;
	BYTE *frame;
	int bytes_used;
	gboolean keyframe;
} VidBuff;

/*global variables used in guvcview*/
struct GLOBAL
{

	__MUTEX_TYPE mutex;    //global struct mutex
	__MUTEX_TYPE file_mutex; //video file mutex
	__COND_TYPE  IO_cond;      //IO thread semaphore

	VidBuff *videoBuff;    //video Buffer
	int video_buff_size;   //size in frames of video buffer

	char *videodevice;     // video device (def. /dev/video0)
	char *confPath;        //configuration file path
	char *vidfile;         //video filename passed through argument options with -n
	char *WVcaption;       //video preview title bar caption
	//char *imageinc_str;    //label for File inc
	//char *vidinc_str;      //label for File inc
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

