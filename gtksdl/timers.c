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

#include <SDL/SDL.h>
/* support for internationalization - i18n */
#include <glib/gi18n.h>
#include <glib/gprintf.h>
#include <gtk/gtk.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/statfs.h>

#include "globals.h"
#include "guvcview.h"
#include "close.h"

#define __VMUTEX &videoIn->mutex

/* called for timed shutdown (from video thread)*/
gboolean
shutd_timer(gpointer data)
{
    /*stop video capture*/
    shutd (0, data);
    
    return (FALSE);/*destroys the timer*/
}

/* called by video capture from start timer */
gboolean
timer_callback(gpointer data)
{
    return (FALSE);/*destroys the timer*/
}

/*called by timed capture [-c seconds] command line option*/
gboolean
Image_capture_timer(gpointer data)
{
    return (TRUE);/*keep the timer*/
}

/* called by fps counter every 2 sec */
gboolean 
FpsCount_callback(gpointer data)
{
    struct ALL_DATA * all_data = (struct ALL_DATA *) data;
    struct GLOBAL *global = all_data->global;
    
    global->DispFps = (double) global->frmCount / 2;

    if (global->FpsCount>0) 
        return(TRUE); /*keeps the timer*/
    else 
    {
        if(!global->no_display)
        {
            g_snprintf(global->WVcaption,13,"LeopardVideo");
            SDL_WM_SetCaption(global->WVcaption, NULL);
        }
        return (FALSE);/*destroys the timer*/
    }
}

/*
 * Not a timer callback
 * Regular function to determine if enought free space is available
 * returns TRUE if still enough free space left on disk
 * FALSE otherwise
 */
gboolean
DiskSupervisor(gpointer data)
{
    return (TRUE); /* still have enough free space on disk */
}

/* called by video capture every 10 sec for checking disk free space*/
gboolean 
FreeDiskCheck_timer(gpointer data)
{
    return (FALSE);/*destroys the timer*/
}

