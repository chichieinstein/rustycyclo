#include "../include/spectral_analyzer.cuh"
#include <cufft.h>
#include <iostream>
// #include <omp.h>
#include <chrono>
#include <cmath>
#include <complex.h>
#include <math.h>
#include <iomanip>
#include <stdio.h>
// #include "pybind11/include/pybind11/pybind11.h"
// #include "pybind11/include/pybind11/numpy.h"
using std::cout;
using std::endl;
//using std::cyl_bessel_i;
const float pi = acos(-1);
// namespace py = pybind11;

void __global__ create_matrix_gpu(cufftComplex *in, cufftComplex *out, int N, int Np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx < N) && (idy < Np))
    {
        out[idx * Np + idy] = in[idx + idy];
    }
}

void __global__ make_conj(cufftComplex *in, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        in[id] = make_cuComplex(in[id].x, -in[id].y);
}

void create_batch_matrix_gpu(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
{
    int jump{(N + Np) / 2};
    int ind;
    dim3 dimBlock(32, 32);
    dim3 dimGrid(N, Np);
    for(ind=0; ind < BATCH; ind++)
        create_matrix_gpu<<<dimGrid, dimBlock>>>(input + ind*jump, output + ind*N*Np, N, Np);
}

void __global__ fft_shift(bool stride, cufftComplex *inp, int N, int Np)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int sign;
    if (id < N*Np)
    {
        if (stride)
        {
            sign = 1 - 2 * (id % 2);
        }
        else
        {
            int row_pos = (int)id / Np;
            sign = 1 - 2 * (row_pos % 2);
        }
        inp[id] = make_cuComplex(inp[id].x * static_cast<float>(sign), inp[id].y * static_cast<float>(sign));
    }
}

void __global__ vect_multiply(cufftComplex *in, cufftComplex *vec, bool row_wise, bool conj, int N, int Np)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_ind;
    if (id < N*Np)
    {
        if (row_wise)
            vec_ind = id % Np;
        else
            vec_ind = id / Np;

        if (conj)
            in[id] =  make_cuComplex(in[id].x * vec[vec_ind].x + in[id].y * vec[vec_ind].y, -in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x);
        else
        {
            //printf("Real part of input : %f, Imaginary part of input : %f, real part of vec : %f, imag part of vec : %f , prod_real : %f, prod_imag : %f\n", in[id].x, in[id].y, vec[vec_ind].x, vec[vec_ind].y, in[id].x * vec[vec_ind].x - in[id].y * vec[vec_ind].y, in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x );
            in[id] = make_cuComplex(in[id].x * vec[vec_ind].x - in[id].y * vec[vec_ind].y, in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x);
        }
    }
}

void __global__ mat_multiply(cufftComplex *left, cufftComplex *right, int N, int Np)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N*Np)
        right[id] = make_cuComplex(left[id].x * right[id].x - left[id].y * right[id].y, left[id].x * right[id].y + left[id].y * right[id].x);
}

void __global__ average(cufftComplex *in, float *out, int N, int Np, int BATCH)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N*Np)
    {
        for (int idx = 0; idx < BATCH; idx ++)
            out[id] += hypotf(in[idx*N*Np + id].x, in[idx*N*Np + id].y);
        out[id] = out[id] / static_cast<float>(BATCH);
    }
}

void __global__ set_zero(float *in, int max_size)
{
    int id          = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < max_size)
    {
        in[id] = static_cast<float>(0);
    }
}

void __global__ reductor(float *in, float *out, int N, int Np, int max_size)
{
    int id          = blockIdx.x * blockDim.x + threadIdx.x;
    //int max_size    = 2 * N - (Np / 2);
    if (id < max_size)
    {
        int reduced_index = (id < N) ? id : id - N + (Np / 2);
        // cycles[id]        = (id < N) ? Q[reduced_index] + K[0] : Q[reduced_index] + K[Np-1];
        int quotient      = (2 * reduced_index) / Np;
        if (id < N)
        {
            for (int new_ind=0; new_ind <= quotient; new_ind++)
            {
                int xind  =  id - (new_ind * Np) / 2;
                out[id] +=  in[xind*Np + new_ind];
            }
        }
        else
        {
            int counter  = (Np - 1) - quotient;
            for (int new_ind=0; new_ind < counter; new_ind++)
            {
                int xind  = reduced_index + (new_ind * Np) / 2;
                out[id]  += in[xind*Np + Np - 1 - new_ind];
            }
        }

    }
}

void create_batched_center(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
{
    int jump_mid{Np / 2};
    int jump_start{(N+Np) / 2};
    for (int id=0; id < BATCH; id++)
        cudaMemcpy(output + id * N, input + (id * jump_start + jump_mid), sizeof(cufftComplex)*N, cudaMemcpyDeviceToDevice);
}


ssca_cuda::ssca_cuda(complex<float>* k1, complex<float>* k2, complex<float>* e_mat, int Nval, int Npval, int BATCHval)
{
    N=Nval;
    Np=Npval;
    BATCH=BATCHval;
    rank = 1;
    istride_1 = 1;
    idist_1 = Np;
    batch_1 = N;
    ostride_1 = 1;
    odist_1 = Np;
    istride_2 = Np;
    idist_2 = 1;
    batch_2 = Np;
    odist_2 = 1;
    ostride_2 = Np;
    n_1 = new int [1];
    n_2 = new int [1];
    *n_1 = Np;
    *n_2 = N;
    inembed_1 = n_1;
    onembed_1 = n_1;
    inembed_2 = n_2;
    onembed_2 = n_2;
    cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
    cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    cudaMalloc((void**) &kaiser_1, sizeof(cufftComplex)*Np);
    cudaMalloc((void**) &kaiser_2, sizeof(cufftComplex)*N);
    cudaMalloc((void**) &exp_mat, sizeof(cufftComplex)*N*Np);
    cudaMemcpy(exp_mat, e_mat, sizeof(cufftComplex)*N*Np, cudaMemcpyHostToDevice);
    cudaMemcpy(kaiser_1, k1, sizeof(cufftComplex)*Np, cudaMemcpyHostToDevice);
    cudaMemcpy(kaiser_2, k2, sizeof(cufftComplex)*N, cudaMemcpyHostToDevice);

    int size = (BATCHval + 1) * ((Nval + Npval) / 2);
    cudaMalloc((void**)&in_buffer, sizeof(cufftComplex)*size);
    cudaMalloc((void**)&ssca_buffer, sizeof(cufftComplex)*N*Np*BATCH);
    cudaMalloc((void**)&ssca_out_buffer, sizeof(cufftComplex)*N*Np*BATCH);
    cudaMalloc((void**)&out_gpu, sizeof(float)*N*Np);
    cudaMalloc((void**)&center_batch_buffer, sizeof(cufftComplex)*N*BATCH);
    /*#pragma omp parallel for private(ind)*/

    // cudaMalloc((void**) &exp_mat_gpu, sizeof(cufftComplex)*N*Np);
    // cudaMalloc((void**) &kaiser_1_gpu, sizeof(cufftComplex)*Np);
    // cudaMalloc((void**) &kaiser_2_gpu, sizeof(cufftComplex)*N);
    cudaMalloc((void**) &inter_gpu, sizeof(cufftComplex)*N*Np);
    // copy_arrays<<<N, Np>>>(exp_mat, exp_mat_gpu, N*Np);
    // copy_arrays<<<Np, 1>>>(kaiser_1, kaiser_1_gpu, Np);
    // copy_arrays<<<N, 1>>>(kaiser_2, kaiser_2_gpu, N);
}

void ssca_cuda::cyclo_gram(cufftComplex *input_left, cufftComplex *input_right, cufftComplex *output, bool conj, bool uncommented)
{
    int i;
    for (i=0; i<BATCH; i++)
    {   
    cudaMemcpy(inter_gpu, input_left + i*N*Np, sizeof(cufftComplex)*N*Np, cudaMemcpyDeviceToDevice);
    
    if (uncommented)
        vect_multiply<<<N, Np>>>(inter_gpu, kaiser_1, true, false, N, Np);
    
    auto err_1 = cudaGetLastError();
    // cout << "Vect_multiply 1 error : " << cudaGetErrorString(err_1) << endl;

    fft_shift<<<N, Np>>>(true, inter_gpu, N, Np);
    auto err_2 = cudaGetLastError();
    // cout << "FFTShift 1 error : " << cudaGetErrorString(err_2) << endl;
    
    auto arr = cufftExecC2C(plan_1, inter_gpu, inter_gpu, CUFFT_FORWARD);
    auto err_3 = cudaGetLastError();
    // cout << "FFT 1 error : " << cudaGetErrorString(err_3) << endl;
    
    mat_multiply<<<N, Np>>>(exp_mat, inter_gpu, N, Np);
    auto err_4 = cudaGetLastError();
    //cout << "Mat multiply error : " << cudaGetErrorString(err_4) << endl;

    vect_multiply<<<N, Np>>>(inter_gpu, kaiser_2, false, false, N, Np);
    auto err_5 = cudaGetLastError();
    //cout << "Vec multiply 2 error : " << cudaGetErrorString(err_5) << endl;

    vect_multiply<<<N, Np>>>(inter_gpu, input_right + i*N, false, conj, N, Np);
    auto err_6 = cudaGetLastError();
    //cout << "Vec multiply 2 error : " << cudaGetErrorString(err_6) << endl;

    fft_shift<<<N, Np>>>(false, inter_gpu, N, Np);
    auto err_7 = cudaGetLastError();
    //cout << "FFTShift 2 error : " << cudaGetErrorString(err_7) << endl;

    auto arr_2 = cufftExecC2C(plan_2, inter_gpu, inter_gpu, CUFFT_FORWARD);
    auto err_8 = cudaGetLastError();
    //cout << "FFT 2 error : " << cudaGetErrorString(err_8) << endl;

    cudaMemcpy(output + (i*N*Np), inter_gpu, sizeof(cufftComplex)*N*Np, cudaMemcpyDeviceToDevice);
    }
}

ssca_cuda::~ssca_cuda()
{
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    delete [] n_1;
    delete [] n_2;
    cudaFree(exp_mat);
    cudaFree(kaiser_1);
    cudaFree(kaiser_2);
    cudaFree(inter_gpu);
    
    cudaFree(ssca_buffer);
    cudaFree(ssca_out_buffer);
    cudaFree(out_gpu);
    cudaFree(center_batch_buffer);
}
