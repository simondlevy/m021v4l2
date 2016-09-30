/*
   libm021v4l2.c - Python extension for Leopard Imaging M021 camera on Linux.
  
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


#include "Python.h"
#include "numpy/arrayobject.h"
#include "m021_v4l2.h"
#include "m021_thread_support.h"

#include <stdio.h>

// XXX support just one camera for now
static m021_thread_data_t thread_data;

static PyObject * init (PyObject * dummy, PyObject * args)
{
    PyObject * obj = NULL;
    PyArrayObject * arr = NULL;

    int bcorrect, gcorrect, rcorrect;

    if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &obj, &bcorrect, &gcorrect, &rcorrect))
        return NULL;

    arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_INOUT_ARRAY);

    if (arr == NULL) {
        PyArray_XDECREF_ERR(arr);
        return NULL;
    }

    int rows = arr->dimensions[0];
    int cols = arr->dimensions[1];

    /*
    for (int i=0; i<arr->dimensions[0]; ++i) {
        for (int j=0; j<arr->dimensions[1]; ++j) {
            for (int k=0; k<arr->dimensions[2]; ++k) {
                uint8_t *v = (uint8_t*)PyArray_GETPTR3(arr, i, j, k);
                *v = 100;
            }
        }
    }
    */

    Py_DECREF(arr);

    //m021_thread_start(&thread_data, rows, cols, bytes, bcorrect, gcorrect, rcorrect);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject * acquire (PyObject * dummy, PyObject * args)
{
   Py_INCREF(Py_None);
    return Py_None;
}

static struct PyMethodDef methods[] = {
    {"init", init, METH_VARARGS, NULL},
    {"acquire", acquire, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initlibm021v4l2 (void)
{
    (void)Py_InitModule("libm021v4l2", methods);
    import_array();
}
