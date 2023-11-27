#pragma once
#include <cufft.h>
// #include <omp.h>
#include <complex.h>
// #include "pybind11/include/pybind11/pybind11.h"
// #include "pybind11/include/pybind11/numpy.h"
using std::complex;

// namespace py = pybind11;

void __global__ create_matrix_gpu(cufftComplex*, cufftComplex*, int, int);

void __global__ make_conj(cufftComplex*, size_t);

void create_batch_matrix_gpu(cufftComplex*, cufftComplex*, int, int, int);

void __global__ fft_shift(bool, cufftComplex*, int, int);

void __global__ vect_multiply(cufftComplex*, cufftComplex*, bool, bool, int, int);

void __global__ mat_multiply(cufftComplex*, cufftComplex*, int, int);

void __global__ average(cufftComplex*, float*, int, int, int);

void __global__ reductor(float*, float*, int, int, int);

void __global__ set_zero(float*, int);

void create_batched_center(cufftComplex*, cufftComplex*, int, int, int);

class ssca_cuda
{
    public:
    int N;
    int Np;
    int BATCH;
    cufftHandle plan_1;
    cufftHandle plan_2;
    int rank;
    int istride_1;
    int idist_1;
    int batch_1;
    int ostride_1;
    int odist_1;
    int istride_2;
    int idist_2;
    int batch_2;
    int odist_2;
    int ostride_2;
    int *n_1;
    int *n_2;
    int *inembed_1;
    int *onembed_1;
    int *inembed_2;
    int *onembed_2;
    cufftComplex *kaiser_1;
    cufftComplex *kaiser_2;
    cufftComplex *exp_mat;
    cufftComplex *inter_gpu;

    cufftComplex* in_buffer;
    cufftComplex* ssca_buffer;
    cufftComplex* ssca_out_buffer;
    float* out_gpu;
    cufftComplex* center_batch_buffer;

    
    ssca_cuda(complex<float>*, complex<float>*, complex<float>*, int, int, int);
    void cyclo_gram(cufftComplex*, cufftComplex*, cufftComplex*, bool, bool);
    void process(cufftComplex*, cufftComplex*, cufftComplex*, bool, bool);
    ~ssca_cuda();
};

