#ifndef GUVCVIEW_H
#define GUVCVIEW_H

#include <gtk/gtk.h>
#include <linux/videodev2.h>
#include <inttypes.h>


#include "../m021_v4l2.h"

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

#endif
