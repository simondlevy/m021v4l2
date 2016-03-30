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


#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include "ms_time.h"

/*------------------------------ get time ------------------------------------*/
/*in miliseconds*/
DWORD ms_time (void)
{
	GTimeVal *tod;
	tod = g_new0(GTimeVal, 1);
	g_get_current_time(tod);
	DWORD mst = (DWORD) tod->tv_sec * 1000 + (DWORD) tod->tv_usec / 1000;
	g_free(tod);
	return (mst);
}
/*in microseconds*/
ULLONG us_time(void)
{
	GTimeVal *tod;
	tod = g_new0(GTimeVal, 1);
	g_get_current_time(tod);
	ULLONG ust = (DWORD) tod->tv_sec * G_USEC_PER_SEC + (DWORD) tod->tv_usec;
	g_free(tod);
	return (ust);
}

/*REAL TIME CLOCK*/
/*in nanoseconds*/
ULLONG ns_time (void)
{
	static struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return ((ULLONG) ts.tv_sec * G_NSEC_PER_SEC + (ULLONG) ts.tv_nsec);
}

/*MONOTONIC CLOCK*/
/*in nanosec*/
UINT64 ns_time_monotonic()
{
	static struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((UINT64) ts.tv_sec * G_NSEC_PER_SEC + (ULLONG) ts.tv_nsec);
}

//sleep for given time in ms
void sleep_ms(int ms_time)
{
	gulong sleep_us = ms_time *1000; /*convert to microseconds*/
	g_usleep( sleep_us );/*sleep for sleep_ms ms*/
}

/*wait on cond by sleeping for n_loops of sleep_ms ms (test var==val every loop)*/
/*return remaining number of loops (if 0 then a stall occurred)              */
int wait_ms(gboolean* var, gboolean val, __MUTEX_TYPE *mutex, int ms_time, int n_loops)
{
	int n=n_loops;
	__LOCK_MUTEX(mutex);
		while( (*var!=val) && ( n > 0 ) ) /*wait at max (n_loops*sleep_ms) ms */
		{
			__UNLOCK_MUTEX(mutex);
			n--;
			sleep_ms( ms_time );/*sleep for sleep_ms ms*/
			__LOCK_MUTEX(mutex);
		};
	__UNLOCK_MUTEX(mutex);
	return (n);
}

