#include "Python.h"
#include "numpy/arrayobject.h"

static PyObject* acquire (PyObject *dummy, PyObject *args)
{
    PyObject * obj = NULL;
    PyArrayObject * arr = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &obj)) return NULL;

    arr = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_UINT8, NPY_INOUT_ARRAY);

    if (arr == NULL) {
        PyArray_XDECREF_ERR(arr);
        return NULL;
    }


    for (int i=0; i<arr->dimensions[0]; ++i) {
        for (int j=0; j<arr->dimensions[1]; ++j) {
            for (int k=0; k<arr->dimensions[2]; ++k) {
                uint8_t *v = (uint8_t*)PyArray_GETPTR3(arr, i, j, k);
                *v = 100;
            }
        }
    }

    Py_DECREF(arr);
    Py_INCREF(Py_None);
    return Py_None;
}

static struct PyMethodDef methods[] = {
    {"acquire", acquire, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC initlibm021v4l2 (void)
{
    (void)Py_InitModule("libm021v4l2", methods);
    import_array();
}
