#include <unistd.h>
#include <SDL/SDL.h>
#include <glib/gi18n.h>
#include <glib.h>
#include <pthread.h>
#include <gtk/gtk.h>
#include <portaudio.h>

#include "globals.h"
#include "ms_time.h"
#include "close.h"

#define __VMUTEX &videoIn->common.mutex

void
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
