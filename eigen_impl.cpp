#include <Eigen/Core>
#include <iostream>

#include <Python.h> // for Py_ssize_t

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> NumpyVecF;

static Eigen::VectorXf buf2vecf(float *mem, Py_ssize_t n)
{
    return Eigen::Map<NumpyVecF>(mem, n);
}

static void vecf2buf(const Eigen::VectorXf& vec, float *mem)
{
    Eigen::Map<NumpyVecF>(mem, vec.size()) = vec;
}


typedef Eigen::Matrix<int, Eigen::Dynamic, 1> NumpyVecS;

static Eigen::VectorXi buf2vecs(int *mem, Py_ssize_t n)
{
    return Eigen::Map<NumpyVecS>(mem, n);
}

static void vecs2buf(const Eigen::VectorXi& vec, int *mem)
{
    Eigen::Map<NumpyVecS>(mem, vec.size()) = vec;
}


// In Python, the default is row-major (C) while in Eigen it's ColMajor (F).
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> NumpyMatF;

static Eigen::MatrixXf buf2matf(float *mem, Py_ssize_t h, Py_ssize_t w)
{
    // This does the conversion, so very likely makes a copy.
    return Eigen::Map<NumpyMatF>(mem, h, w);
}

static void matf2buf(const Eigen::MatrixXf& mat, float *mem)
{
    Eigen::Map<NumpyMatF>(mem, mat.rows(), mat.cols()) = mat;
}

