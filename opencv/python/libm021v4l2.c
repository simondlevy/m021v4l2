#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject* acquire (PyObject *dummy, PyObject *args)
{
    PyObject *out=NULL;
    PyArrayObject *oarr=NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &out)) return NULL;

    oarr = (PyArrayObject*)PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (oarr == NULL) goto fail;

    /*vv* code that makes use of arguments *vv*/

    for (int i=0; i<oarr->dimensions[0]; ++i) {
        for (int j=0; j<oarr->dimensions[1]; ++j) {
            double *v = (double*)PyArray_GETPTR2(oarr, i, j);
            *v = *v * 2;
        }
    }
    /*^^* code that makes use of arguments *^^*/

    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

fail:
    PyArray_XDECREF_ERR(oarr);
    return NULL;
}

static struct PyMethodDef methods[] = {
    {"acquire", acquire, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
initlibm021v4l2 (void)
{
    (void)Py_InitModule("libm021v4l2", methods);
    import_array();
}
