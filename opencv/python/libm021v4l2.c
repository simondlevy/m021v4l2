#include "Python.h"
#include "numpy/arrayobject.h"

#include <stdio.h>

static PyObject* example (PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyArrayObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2,
                &PyArray_Type, &out)) return NULL;

    arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL) goto fail;
    oarr = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (oarr == NULL) goto fail;

    /*vv* code that makes use of arguments *vv*/

    int nd = PyArray_NDIM(arr1);   //number of dimensions
    npy_intp *shape = PyArray_DIMS(arr1);  // npy_intp array of length nd showing length in each dim.
    for (int i=0; i<nd; ++i)
        printf("%d ", shape[i]);
    printf("\n");

    for (int i=0; i<arr2->nd; ++i)
        printf("%d ", arr2->dimensions[i]);
    printf("\n");

    for (int i=0; i<oarr->dimensions[0]; ++i) {
        for (int j=0; j<oarr->dimensions[1]; ++j) {
            double *v = (double*)PyArray_GETPTR2(oarr, i, j);
            *v = *v * 2;
        }
    }
    /*^^* code that makes use of arguments *^^*/

    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    PyArray_XDECREF_ERR(oarr);
    return NULL;
}

static struct PyMethodDef methods[] = {
    {"example", example, METH_VARARGS, "descript of example"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initlibm021v4l2 (void)
{
    (void)Py_InitModule("libm021v4l2", methods);
    import_array();
}
