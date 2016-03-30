/*******************************************************************************#

#           guvcview              http://guvcview.sourceforge.net               #
#                                                                               #
#           Paulo Assis <pj.assis@gmail.com>                                    #
#           Nobuhiro Iwamatsu <iwamatsu@nigauri.org>                            #
#                             Add UYVY color support(Macbook iSight)            #
#           Flemming Frandsen <dren.dk@gmail.com>                               #
#                             Add VU meter OSD                                  #
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

#include <portaudio.h>
#include <SDL/SDL.h>
#include <glib.h>
#include <glib/gprintf.h>
#include <gtk/gtk.h>

#include "defs.h"
#include "video.h"
#include "guvcview.h"
#include "globals.h"
#include "ms_time.h"
#include "timers.h"

#define __VMUTEX &videoIn->common.mutex
#define __GMUTEX &global->mutex

static Uint32 SDL_VIDEO_Flags =
        SDL_ANYFORMAT | SDL_RESIZABLE;

static const SDL_VideoInfo *info;

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
void *main_loop(void *data)
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



