#include "../include/spectral_analyzer.cuh"
#include <cufft.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <complex.h>
#include <math.h>
#include <iomanip>
#include <stdio.h>

using std::cout;
using std::endl;
const float pi = acos(-1);
void __global__ make_conj(cufftComplex *in, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        in[id] = make_cuComplex(in[id].x, -in[id].y);
}


void __global__ average(cufftComplex *in, float *out, int N, int Np, int BATCH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int id = idy * Np + idx;
    if ((idx < Np) && (idy < N))
    {
        for (int idz = 0; idz < BATCH; idz++)
            out[id] += hypotf(in[idz * N * Np + id].x, in[idz * N * Np + id].y);
        out[id] = out[id] / static_cast<float>(BATCH);
    }
}

void __global__ batch_average(cufftComplex *in, float *out, int N, int Np, int BATCH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int id2d = idy * Np + idx;

    if ((idx < Np) && (idy < N) && (idz < BATCH))
    {
        atomicAdd(&out[id2d], hypotf(in[idz * N * Np + id2d].x / BATCH, in[idz * N * Np + id2d].y / BATCH));
    }
}

void __global__ set_zero(float *in, int max_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < max_size)
    {
        in[id] = static_cast<float>(0);
    }
}

void __global__ reductor(float *in, float *out, int N, int Np, int max_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // int max_size    = 2 * N - (Np / 2);
    if (id < max_size)
    {
        int reduced_index = (id < N) ? id : id - N + (Np / 2);
        // cycles[id]        = (id < N) ? Q[reduced_index] + K[0] : Q[reduced_index] + K[Np-1];
        int quotient = (2 * reduced_index) / Np;
        if (id < N)
        {
            for (int new_ind = 0; new_ind <= quotient; new_ind++)
            {
                int xind = id - (new_ind * Np) / 2;
                out[id] += in[xind * Np + new_ind];
            }
        }
        else
        {
            int counter = (Np - 1) - quotient;
            for (int new_ind = 0; new_ind < counter; new_ind++)
            {
                int xind = reduced_index + (new_ind * Np) / 2;
                out[id] += in[xind * Np + Np - 1 - new_ind];
            }
        }
    }
}
// Called with stride = true
void __global__ mat_mul_fft_shift_batch_reshape(cufftComplex *window, cufftComplex *input, cufftComplex *out, int N, int Np, int BATCH)
{
    int jump{(N + Np)};
    int sign;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int id3d = idz * N * Np + idy * Np + idx;
    int id = idy * Np + idx;

    int input_id = idz * jump + idx + idy;
    int window_id = idx;

    if ((idx < Np) && (idy < N) && (idz < BATCH))
    {
        sign = 1 - 2 * (id % 2);
        out[id3d] = make_cuComplex((input[input_id].x * window[window_id].x - input[input_id].y * window[window_id].y) * static_cast<float>(sign), (input[input_id].x * window[window_id].y + input[input_id].y * window[window_id].x) * static_cast<float>(sign));
    }
}

// Called with stride = false
// left_mat should have dimensions N*Np and contains the product of exp_mat and kaiser_2
void __global__ mat_vec_multiply_fft_shift_batch_center(cufftComplex *left_mat, cufftComplex* input, cufftComplex *right, bool conj, int N, int Np, int BATCH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // int id1d = idz * N + idy;
    int id2d = idy * Np + idx;
    int id3d = idz * N * Np + id2d;
    int sign;
    // int id = idz * N * Np + idy * Np + idx;

    int jump{(N + Np)};
    int jump_mid{Np / 2};

    int id_jump = idz * jump + jump_mid + idy;

    cufftComplex intermediate;
    if ((idx < Np) && (idy < N) && (idz < BATCH))
    {
        int row_pos = (int)id2d / Np;
        sign = 1 - 2 * (row_pos % 2);
        if (conj)
            intermediate = make_cuComplex(left_mat[id2d].x * input[id_jump].x + left_mat[id2d].y * input[id_jump].y, -left_mat[id2d].x * input[id_jump].y + left_mat[id2d].y * input[id_jump].x);
        else
        {
            intermediate = make_cuComplex(left_mat[id2d].x * input[id_jump].x - left_mat[id2d].y * input[id_jump].y, left_mat[id2d].x * input[id_jump].y + left_mat[id2d].y * input[id_jump].x);
        }
        right[id3d] = make_cuComplex((intermediate.x * right[id3d].x - intermediate.y * right[id3d].y) * static_cast<float>(sign), (intermediate.x * right[id3d].y + intermediate.y * right[id3d].x) * static_cast<float>(sign));
    }
}

// void __global__ vect_multiply_batch(cufftComplex *in, cufftComplex *vec, bool conj, int N, int Np, int BATCH)
// {
//     // int id = blockIdx.x * blockDim.x + threadIdx.x;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;

//     int id1d = idz * N + idy;
//     int id2d = idy * Np + idx;
//     int id3d = idz * N * Np + id2d;
//     // int vec_ind;
//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         if (conj)
//             in[id3d] = make_cuComplex(in[id3d].x * vec[id1d].x + in[id3d].y * vec[id1d].y, -in[id3d].x * vec[id1d].y + in[id3d].y * vec[id1d].x);
//         else
//         {
//             in[id3d] = make_cuComplex(in[id3d].x * vec[id1d].x - in[id3d].y * vec[id1d].y, in[id3d].x * vec[id1d].y + in[id3d].y * vec[id1d].x);
//         }
//     }
// }

// void create_batched_center(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
// {
//     int jump_mid{Np / 2};
//     int jump_start{(N + Np) / 2}; // bool conj;
//     // complex<float> *load_buffer;
//     for (int id = 0; id < BATCH; id++)
//         cudaMemcpy(output + id * N, input + (id * jump_start + jump_mid), sizeof(cufftComplex) * N, cudaMemcpyDeviceToDevice);
// }

ssca_cuda::ssca_cuda(complex<float> *k1, complex<float> *e_mat, int Nval, int Npval, int size_val)
{
    N = Nval;
    Np = Npval;
    BATCH = size_val / (Nval + Npval);
    size = size_val;
    rank = 1;
    istride_1 = 1;
    idist_1 = Np;
    batch_1 = N * BATCH;
    ostride_1 = 1;
    odist_1 = Np;
    istride_2 = Np;
    idist_2 = 1;
    batch_2 = Np;
    odist_2 = 1;
    ostride_2 = Np;
    // istride_2 = 1;
    // idist_2 = N;
    // batch_2 = Np*BATCH;
    // odist_2 = N;
    // ostride_2 = 1;
    n_1 = new int[1];
    n_2 = new int[1];
    *n_1 = Np;
    *n_2 = N;
    inembed_1 = n_1;
    onembed_1 = n_1;
    inembed_2 = n_2;
    onembed_2 = n_2;

    streams = new cudaStream_t[BATCH];
    // plans_2 = new cufftHandle[BATCH];
    events = new cudaEvent_t[BATCH];

    cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
    cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    // for (int i = 0; i < BATCH; i++)
    // {
    //     cudaStreamCreate(&streams[i]);
    //     cudaEventCreate(&events[i]);
    //     cufftPlanMany(&plans_2[i], rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    // }
    
    cudaMalloc((void **)&kaiser_1, sizeof(cufftComplex) * Np);
    cudaMalloc((void **)&exp_mat, sizeof(cufftComplex) * N * Np);

    cudaMemcpy(exp_mat, e_mat, sizeof(cufftComplex) * N * Np, cudaMemcpyHostToDevice);
    cudaMemcpy(kaiser_1, k1, sizeof(cufftComplex) * Np, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&inp_buffer, sizeof(cufftComplex) * size);
    cudaMalloc((void **)&inter_gpu, sizeof(cufftComplex) * N * Np * BATCH);
    // cudaMalloc((void **)&inter_inter_gpu, sizeof(cufftComplex) * N * Np * BATCH);
    // cudaMalloc((void **)&inter_center_gpu, sizeof(cufftComplex) * N * BATCH);
    cudaMalloc((void **)&output_buffer, sizeof(float) * N * Np);
}

void ssca_cuda::cyclo_gram(cufftComplex *input, float *output, bool conj)
{
    dim3 dimBlockMatrixCreation(16, 16, 4);
    dim3 dimGridMatrixCreation(Np / 16, N / 16, BATCH / 4);

    dim3 dimBlockReshape(32, 32);
    dim3 dimGridReshape(Np / 32, N / 32);

    cudaMemcpy(inp_buffer, input, sizeof(cufftComplex) * size, cudaMemcpyHostToDevice);
    mat_mul_fft_shift_batch_reshape<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(kaiser_1, inp_buffer, inter_gpu, N, Np, BATCH);
    auto error = cudaGetLastError();
    // cout << cudaGetErrorString(error) << endl;
    auto arr = cufftExecC2C(plan_1, inter_gpu, inter_gpu, CUFFT_FORWARD);
    auto error_2 = cudaGetLastError();
    // cout << cudaGetErrorString(error_2) << endl;
    mat_vec_multiply_fft_shift_batch_center<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(exp_mat, inp_buffer, inter_gpu, conj, N, Np, BATCH);
    auto error_3 = cudaGetLastError();
    // cout << cudaGetErrorString(error_3) << endl;
    for (int i = 0; i < BATCH; i++)
    {
        // auto err = cufftExecC2C(plans_2[i], inter_gpu + i*N*Np, inter_gpu + i*N*Np, CUFFT_FORWARD);
        auto err = cufftExecC2C(plan_2, inter_gpu + i*N*Np, inter_gpu + i*N*Np, CUFFT_FORWARD);
        auto err_3 = cudaGetLastError();
        // cout << cudaGetErrorString(error_3) << endl;
        // cudaEventRecord(events[i], streams[i]);
    }

    // for (int i = 0; i < BATCH; i++)
    // {
    //     cudaStreamWaitEvent(0, events[i], 0);
    // }
    // batch_average<<<dimBlockMatrixCreation, dimGridMatrixCreation>>>(inter_gpu, output_buffer, N, Np, BATCH);
    average<<<dimBlockReshape, dimGridReshape>>>(inter_gpu, output_buffer, N, Np, BATCH);
    // auto error_5 = cudaGetLastError();
    // cout << cudaGetErrorString(error_5) << endl;
    cudaMemcpy(output, output_buffer, sizeof(float) * N * Np, cudaMemcpyDeviceToDevice);
}
ssca_cuda::~ssca_cuda()
{
    for (int i = 0; i < BATCH; i++)
    {
        // cufftDestroy(plans_2[i]);
        // cufftDestroy(plans_1[i]);
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
    }
    cufftDestroy(plan_1);
    cufftDestroy(plan_2);
    delete[] n_1;
    delete[] n_2;

    // delete[] streams;
    // delete[] events;
    // // delete[] plans_1;
    // delete[] plans_2;
    cudaFree(exp_mat);
    cudaFree(kaiser_1);
    // cudaFree(kaiser_2);
    // cudaFree(exp_mat_gpu);
    cudaFree(inter_gpu);
    // cudaFree(inter_inter_gpu);
    cudaFree(inp_buffer);
    cudaFree(output_buffer);
    // cudaFree(inter_center_gpu);
    // cudaFree(kaiser_1_gpu);
    // cudaFree(kaiser_2_gpu);
}
