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

#ifndef DEFS_H
#define DEFS_H
#include <inttypes.h>
#include <sys/types.h>

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

/*----------- guvcview version ----------------*/
//#define VERSION ("") /*defined in config.h*/
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

#endif

