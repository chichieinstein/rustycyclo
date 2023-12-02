#pragma once
#include <cufft.h>
#include <complex.h>
using std::complex;

// void __global__ create_matrix_gpu(cufftComplex*, cufftComplex*, int, int);

void __global__ make_conj(cufftComplex*, size_t);

// void create_batch_matrix_gpu(cufftComplex*, cufftComplex*, int, int, int);

// void __global__ fft_shift(bool, cufftComplex*, int, int);

// void __global__ vect_multiply(cufftComplex*, cufftComplex*, bool, bool, int, int);

// void __global__ mat_multiply(cufftComplex*, cufftComplex*, int, int);

void __global__ batch_average(cufftComplex*, float*, int, int, int);

void __global__ reductor(float*, float*, int, int, int);

void __global__ set_zero(float*, int);

// void create_batched_center(cufftComplex*, cufftComplex*, int, int, int);

void __global__ mat_mul_fft_shift_batch_reshape(cufftComplex*, cufftComplex*, cufftComplex*, int, int, int);

void __global__ mat_vec_multiply_fft_shift_batch_center(cufftComplex*, cufftComplex*, cufftComplex*, bool, int, int, int);

class ssca_cuda
{
    public:
    int N;
    int Np;
    int BATCH;
    int size;
    cufftHandle plan_1;
    // cufftHandle plan_2;
    // cufftHandle *plans_1;
    cufftHandle *plans_2;
    cudaStream_t * streams;
    cudaEvent_t * events;
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
    // cufftComplex *kaiser_2;
    cufftComplex *exp_mat;
    cufftComplex *inter_gpu;
    // cufftComplex *inter_inter_gpu;
    cufftComplex *inp_buffer;
    // cufftComplex *inter_center_gpu;

    float* output_buffer;
    
    ssca_cuda(complex<float>*, complex<float>*, int, int, int);
    void cyclo_gram(cufftComplex*, float*, bool);
    ~ssca_cuda();
};

