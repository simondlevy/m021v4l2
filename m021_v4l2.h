/*
   m021_v4l2.h :  Header for LI-USB30-M021 V4L2 code

   Copyright (C) 2016 Simon D. Levy

   This file is part of M021_V4L2.

   M021_V4L2 is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   BreezySTM32 is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with M021_V4L2.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef M021_H
#define M021_H

#include <linux/videodev2.h>
#include <libudev.h>
#include <pthread.h>
#include <stdbool.h>

#define NB_BUFFER 4

#define VDIN_OK                    0
#define VDIN_DEVICE_ERR           -1
#define VDIN_FORMAT_ERR           -2
#define VDIN_REQBUFS_ERR          -3
#define VDIN_ALLOC_ERR            -4
#define VDIN_RESOL_ERR            -5
#define VDIN_FBALLOC_ERR          -6
#define VDIN_UNKNOWN_ERR          -7
#define VDIN_DEQBUFS_ERR          -8
#define VDIN_DECODE_ERR           -9
#define VDIN_QUERYCAP_ERR        -10
#define VDIN_QUERYBUF_ERR        -11
#define VDIN_QBUF_ERR            -12
#define VDIN_MMAP_ERR            -13
#define VDIN_READ_ERR            -14
#define VDIN_STREAMON_ERR        -15
#define VDIN_STREAMOFF_ERR       -16
#define VDIN_DYNCTRL_ERR         -17

typedef struct m021 {

    pthread_mutex_t mutex;

    struct   udev *udev;                  // pointer to a udev struct (lib udev)
    struct   udev_monitor *udev_mon;      // udev monitor
    int      udev_fd;                     // udev monitor file descriptor
    int      fd;                          // device file descriptor
    struct   v4l2_capability cap;         // v4l2 capability struct
    struct   v4l2_format fmt;             // v4l2 formar struct
    struct   v4l2_buffer buf;             // v4l2 buffer struct
    struct   v4l2_requestbuffers rb;      // v4l2 request buffers struct
    struct   v4l2_streamparm streamparm;  // v4l2 stream parameters struct
    void *   mem[NB_BUFFER];              // memory buffers for mmap driver frames
    uint32_t buff_length[NB_BUFFER];      // memory buffers length as set by VIDIOC_QUERYBUF
    uint32_t buff_offset[NB_BUFFER];      // memory buffers offset as set by VIDIOC_QUERYBUF
    int      isstreaming;                 // video stream flag (1- ON  0- OFF)
    int      available_exp[4];            // backward compatible (old v4l2 exposure menu interface)
    int      format;
    int      width;
    int      height;

} m021_t;

typedef struct m021_1280x720 {

    uint8_t tmpbuffer[1280*720*2];     // temp buffer for decoding compressed data
    uint8_t tmpbuffer1[1280*720*3];    // temp buffer for converting bayer16 to bayer8
    uint8_t framebuffer[1280*720*2];   // frame buffer (YUYV)

    m021_t common;

} m021_1280x720_t;

typedef struct m021_800x460 {

    uint8_t tmpbuffer[800*460*2];     // temp buffer for decoding compressed data
    uint8_t tmpbuffer1[800*460*3];    // temp buffer for converting bayer16 to bayer8
    uint8_t framebuffer[800*460*2];   // frame buffer (YUYV)

    m021_t common;

} m021_800x460_t;

typedef struct m021_640x480 {

    uint8_t tmpbuffer[640*480*2];     // temp buffer for decoding compressed data
    uint8_t tmpbuffer1[640*480*3];    // temp buffer for converting bayer16 to bayer8
    uint8_t framebuffer[640*480*2];   // frame buffer (YUYV)

    m021_t common;

} m021_640x480_t;

int m021_1280x720_init(int id, m021_1280x720_t * videoIn);
int m021_1280x720_grab_yuyv(m021_1280x720_t * m021, uint8_t * frame);
int m021_1280x720_grab_bgr(m021_1280x720_t * m021, uint8_t * frame);

int m021_800x460_init(int id, m021_800x460_t * videoIn);
int m021_800x460_grab_yuyv(m021_800x460_t * m021, uint8_t * frame);
int m021_800x460_grab_bgr(m021_800x460_t * m021, uint8_t * frame);

int m021_640x480_init(int id, m021_640x480_t * videoIn);
int m021_640x480_grab_yuyv(m021_640x480_t * m021, uint8_t * frame);
int m021_640x480_grab_bgr(m021_640x480_t * m021, uint8_t * frame);

#endif

