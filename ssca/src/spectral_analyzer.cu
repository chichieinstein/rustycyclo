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

// void __global__ fft_shift(bool stride, cufftComplex *inp, int N, int Np)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int id = idy * Np + idx;
//     int sign;
//     if ((idy < N) && (idx < Np))
//     {
//         if (stride)
//         {
//             sign = 1 - 2 * (id % 2);
//         }
//         else
//         {
//             int row_pos = (int)id / Np;
//             sign = 1 - 2 * (row_pos % 2);
//         }
//         inp[id] = make_cuComplex(inp[id].x * static_cast<float>(sign), inp[id].y * static_cast<float>(sign));
//     }
// }

// void __global__ vect_multiply(cufftComplex *in, cufftComplex *vec, bool row_wise, bool conj, int N, int Np)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.x;
//     int id = idy * Np + idx;
//     int vec_ind;
//     if ((idx < Np) && (idy < N))
//     {
//         if (row_wise)
//             vec_ind = idx;
//         else
//             vec_ind = idy;

//         if (conj)
//             in[id] = make_cuComplex(in[id].x * vec[vec_ind].x + in[id].y * vec[vec_ind].y, -in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x);
//         else
//         {
//             // printf("Real part of input : %f, Imaginary part of input : %f, real part of vec : %f, imag part of vec : %f , prod_real : %f, prod_imag : %f\n", in[id].x, in[id].y, vec[vec_ind].x, vec[vec_ind].y, in[id].x * vec[vec_ind].x - in[id].y * vec[vec_ind].y, in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x );
//             in[id] = make_cuComplex(in[id].x * vec[vec_ind].x - in[id].y * vec[vec_ind].y, in[id].x * vec[vec_ind].y + in[id].y * vec[vec_ind].x);
//         }
//     }
// }

// void __global__ mat_multiply(cufftComplex *left, cufftComplex *right, int N, int Np)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int id = idy * Np + idx;
//     // int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if ((idy < N) && (idx < Np))
//         right[id] = make_cuComplex(left[id].x * right[id].x - left[id].y * right[id].y, left[id].x * right[id].y + left[id].y * right[id].x);
// }

// void __global__ average(cufftComplex *in, float *out, int N, int Np, int BATCH)
// {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id < N * Np)
//     {
//         for (int idx = 0; idx < BATCH; idx++)
//             out[id] += hypotf(in[idx * N * Np + id].x, in[idx * N * Np + id].y);
//         out[id] = out[id] / static_cast<float>(BATCH);
//     }
// }

void __global__ batch_average(cufftComplex *in, float *out, int N, int Np, int BATCH)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int id2d = idy * Np + idx;

    if ((idx < Np) && (idy < N) && (idz < BATCH))
    {
        out[id2d] += hypotf(in[idz * N * Np + id2d].x, in[idz * N * Np + id2d].y);
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

// void __global__ create_matrix_gpu(cufftComplex *in, cufftComplex *out, int N, int Np)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((idx < Np) && (idy < N))
//     {
//         out[idy * Np + idx] = in[idx + idy];
//     }
// }

// void create_batch_matrix_gpu(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
// {
//     int jump{(N + Np) / 2};
//     int ind;
//     dim3 dimBlock(32, 32);
//     dim3 dimGrid(N / 32, Np / 32);
//     for (ind = 0; ind < BATCH; ind++)
//         create_matrix_gpu<<<dimGrid, dimBlock>>>(input + ind * jump, output + ind * N * Np, N, Np);
// }

// void __global__ create_batch_matrix_gpu_3d(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
// {
//     int jump{(N + Np) / 2};
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;

//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         output[idz * Np * N + idy * Np + idx] = input[idz * jump + idx + idy];
//     }
// }

// void __global__ fast_create_center_batch_gpu(cufftComplex *input, cufftComplex *output, int N, int Np, int BATCH)
// {
//     int jump{(N + Np) / 2};
//     int jump_mid{Np / 2};
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if ((idx < N) && (idy < BATCH))
//     {
//         output[idy * N + idx] = input[idy * jump + jump_mid + idx];
//     }
// }

// void __global__ create_center_matrix_gpu(cufftComplex *input, cufftComplex *output, int N, int Np)
// {
//     // int jump{(N + Np) / 2};
//     int jump_mid{Np / 2};

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if ((idx < N))
//     {
//         output[idx] = input[jump_mid + idx];
//     }
// }

// void __global__ mat_multiply_batch(cufftComplex *left, cufftComplex *right, int N, int Np, int BATCH)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;
//     int id = idz * N * Np + idy * Np + idx;
//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         right[id] = make_cuComplex(left[id].x * right[id].x - left[id].y * right[id].y, left[id].x * right[id].y + left[id].y * right[id].x);
//     }
// }

// void __global__ fft_shift_batch(bool stride, cufftComplex *inp, int N, int Np, int BATCH)
// {
//     int sign;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;
//     int id = idy * Np + idx;
//     int id3d = idz * N * Np + id;
//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         if (stride)
//         {
//             sign = 1 - 2 * (id % 2);
//         }
//         else
//         {
//             int row_pos = (int)id / Np;
//             sign = 1 - 2 * (row_pos % 2);
//         }
//         inp[id3d] = make_cuComplex(inp[id3d].x * static_cast<float>(sign), inp[id3d].y * static_cast<float>(sign));
//     }
// }

// void __global__ mat_mul_fft_shift_batch(bool stride, cufftComplex *left, cufftComplex *right, int N, int Np, int BATCH)
// {
//     int sign;
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;
//     int id3d = idz * N * Np + idy * Np + idx;
//     int id = idy * Np + idx;
//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         if (stride)
//         {
//             sign = 1 - 2 * (id % 2);
//         }
//         else
//         {
//             int row_pos = (int)id / Np;
//             sign = 1 - 2 * (row_pos % 2);
//         }
//         right[id3d] = make_cuComplex((left[id3d].x * right[id3d].x - left[id3d].y * right[id3d].y) * static_cast<float>(sign), (left[id3d].x * right[id3d].y + left[id3d].y * right[id3d].x) * static_cast<float>(sign));
//     }
// }

// Called with stride = true
void __global__ mat_mul_fft_shift_batch_reshape(cufftComplex *window, cufftComplex *input, cufftComplex *out, int N, int Np, int BATCH)
{
    int jump{(N + Np) / 2};
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
        // else
        // {
        //     int row_pos = (int)id / Np;
        //     sign = 1 - 2 * (row_pos % 2);
        // }
        out[id3d] = make_cuComplex((input[input_id].x * window[window_id].x - input[input_id].y * window[window_id].y) * static_cast<float>(sign), (input[input_id].x * window[window_id].y + input[input_id].y * window[window_id].x) * static_cast<float>(sign));
    }
}

// // This reshape function has been tested to be correct
// void __global__ reshape(cufftComplex *inp, cufftComplex *output, int nrow, int ncoloumn)
// {
//     __shared__ cufftComplex tile[BLOCKCHANNELS][BLOCKSLICES];
//     int input_x_coord = blockIdx.x * blockDim.x + threadIdx.x;
//     int input_y_coord = blockIdx.y * blockDim.y + threadIdx.y;
//     // int z_coord = blockIdx.z * blockDim.z + threadIdx.z;
//     auto inter = inp + ncoloumn * input_y_coord + input_x_coord;
//     tile[threadIdx.x][threadIdx.y] = *inter;
//     __syncthreads();
//     int output_grid_y_coord = (blockIdx.x * blockDim.x + threadIdx.y) * nrow;
//     int output_grid_x_coord = blockIdx.y * blockDim.y + threadIdx.x;
//     auto outer = output + output_grid_y_coord + output_grid_x_coord;
//     (*outer) = tile[threadIdx.y][threadIdx.x];
// }

// void __global__ reshape(cufftComplex *inp, cufftComplex *output, int nrow, int ncoloumn, int batch)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;

//     int input_id = idz * nrow * ncoloumn + idy * 
// }

// void __global__ mat_vec_multiply_fft_shift_batch(bool stride, cufftComplex *left_mat, cufftComplex *left_vec, cufftComplex *right, bool conj, int N, int Np, int BATCH)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;
//     int idz = blockIdx.z * blockDim.z + threadIdx.z;

//     int id1d = idz * N + idy;
//     int id2d = idy * Np + idx;
//     int id3d = idz * N * Np + id2d;
//     int sign;
//     // int id = idz * N * Np + idy * Np + idx;
//     cufftComplex intermediate;
//     if ((idx < Np) && (idy < N) && (idz < BATCH))
//     {
//         if (stride)
//         {
//             sign = 1 - 2 * (id2d % 2);
//         }
//         else
//         {
//             int row_pos = (int)id2d / Np;
//             sign = 1 - 2 * (row_pos % 2);
//         }
//         if (conj)
//             intermediate = make_cuComplex(left_mat[id3d].x * left_vec[id1d].x + left_mat[id3d].y * left_vec[id1d].y, -left_mat[id3d].x * left_vec[id1d].y + left_mat[id3d].y * left_vec[id1d].x);
//         else
//         {
//             intermediate = make_cuComplex(left_mat[id3d].x * left_vec[id1d].x - left_mat[id3d].y * left_vec[id1d].y, left_mat[id3d].x * left_vec[id1d].y + left_mat[id3d].y * left_vec[id1d].x);
//         }
//         right[id3d] = make_cuComplex((intermediate.x * right[id3d].x - intermediate.y * right[id3d].y) * static_cast<float>(sign), (intermediate.x * right[id3d].y + intermediate.y * right[id3d].x) * static_cast<float>(sign));
//     }
// }

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

    int jump{(N + Np) / 2};
    int jump_mid{Np / 2};

    int id_jump = idz * jump + jump_mid + idy;

    cufftComplex intermediate;
    if ((idx < Np) && (idy < N) && (idz < BATCH))
    {
        // if (stride)
        // {
        //     sign = 1 - 2 * (id2d % 2);
        // }
        // else
        // {
        int row_pos = (int)id2d / Np;
        sign = 1 - 2 * (row_pos % 2);
        // }
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
    BATCH = 2 * size_val / (Nval + Npval);
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
    // // plans_1 = new cufftHandle [BATCH];
    plans_2 = new cufftHandle[BATCH];
    events = new cudaEvent_t[BATCH];

    cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1, onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
    // cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);

    for (int i = 0; i < BATCH; i++)
    {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
        cufftPlanMany(&plans_2[i], rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
        // cufftPlanMany(&plans_2[i], rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
        // cufftSetStream(plans_1[i], streams[i]);
        // cufftSetStream(plans_2[i], streams[i]);
    }
    // cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2, onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
    cudaMalloc((void **)&kaiser_1, sizeof(cufftComplex) * Np * N * BATCH);
    // cudaMalloc((void **)&kaiser_2, sizeof(cufftComplex) * N * Np * BATCH);
    cudaMalloc((void **)&exp_mat, sizeof(cufftComplex) * N * Np * BATCH);
    cudaMemcpy(exp_mat, e_mat, sizeof(cufftComplex) * N * Np, cudaMemcpyHostToDevice);
    // cudaMemcpy(kaiser_1, k1, sizeof(cufftComplex) * Np * N * BATCH, cudaMemcpyHostToDevice);
    cudaMemcpy(kaiser_1, k1, sizeof(cufftComplex) * Np, cudaMemcpyHostToDevice);
    // cudaMemcpy(kaiser_2, k2, sizeof(cufftComplex) * N * N * BATCH, cudaMemcpyHostToDevice);
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

    // dim3 dimBlockCenterCreation(32, 32);
    // dim3 dimGridCenterCreation(N / 32, BATCH / 32);

    // dim3 dimBlockReshape(32, 32);
    // dim3 dimGridReshape(Np / 32, N / 32);

    // dim3 dimBlockReshapeAgain(32, 32);
    // dim3 dimGridReshapeAgain(N / 32, Np / 32);

    cudaMemcpy(inp_buffer, input, sizeof(cufftComplex) * size, cudaMemcpyHostToDevice);
    mat_mul_fft_shift_batch_reshape<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(kaiser_1, inp_buffer, inter_gpu, N, Np, BATCH);
    // create_batch_matrix_gpu_3d<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(inp_buffer, inter_gpu, N, Np, BATCH);
    // fast_create_center_batch_gpu<<<dimGridCenterCreation, dimBlockCenterCreation>>>(inp_buffer, inter_center_gpu, N, Np, BATCH);
    // mat_multiply_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(kaiser_1, inter_gpu, N, Np, BATCH);
    // fft_shift_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(true, inter_gpu, N, Np, BATCH);
    // mat_mul_fft_shift_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(true, kaiser_1, inter_gpu, N, Np, BATCH);
    auto arr = cufftExecC2C(plan_1, inter_gpu, inter_gpu, CUFFT_FORWARD);
    // mat_multiply_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(exp_mat, inter_gpu, N, Np, BATCH);
    // // mat_multiply_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(kaiser_2, inter_gpu, N, Np, BATCH);
    // // TODO : Combine Kaiser 2 and ExpMat into ExpMat.
    // vect_multiply_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(inter_gpu, inter_center_gpu, conj, N, Np, BATCH);
    // mat_vec_multiply_fft_shift_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(false, exp_mat, inter_center_gpu, inter_gpu, conj, N, Np, BATCH);
    mat_vec_multiply_fft_shift_batch_center<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(exp_mat, inp_buffer, inter_gpu, conj, N, Np, BATCH);
    // fft_shift_batch<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(false, inter_gpu, N, Np, BATCH);
    // reshape<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(inter_gpu, inter_inter_gpu, N, Np);
    // auto arr_2 = cufftExecC2C(plan_2, inter_inter_gpu, inter_inter_gpu, CUFFT_FORWARD);
    // reshape<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(inter_inter_gpu, inter_gpu, Np, N);
    for (int i = 0; i < BATCH; i++)
    {
        // reshape<<<dimGridReshape, dimBlockReshape,0,streams[i]>>>(inter_gpu + i*N*Np, inter_inter_gpu + i*N*Np, N, Np);
        // reshape<<<dimGridReshapeAgain, dimBlockReshapeAgain,0,streams[i]>>>(inter_inter_gpu + i*N*Np, inter_gpu + i*N*Np, N, Np);
        auto err = cufftExecC2C(plans_2[i], inter_gpu + i*N*Np, inter_gpu + i*N*Np, CUFFT_FORWARD);
        auto err_3 = cudaGetLastError();
        cudaEventRecord(events[i], streams[i]);
    }

    for (int i = 0; i < BATCH; i++)
    {
        cudaStreamWaitEvent(0, events[i], 0);
    }
    // auto err = cufftExecC2C(plan_2, inter_inter_gpu, inter_inter_gpu, CUFFT_FORWARD);
    batch_average<<<dimGridMatrixCreation, dimBlockMatrixCreation>>>(inter_gpu, output_buffer, N, Np, BATCH);
    cudaMemcpy(output, output_buffer, sizeof(float) * N * Np, cudaMemcpyDeviceToDevice);
}
ssca_cuda::~ssca_cuda()
{
    for (int i = 0; i < BATCH; i++)
    {
        cufftDestroy(plans_2[i]);
        // cufftDestroy(plans_1[i]);
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
    }
    cufftDestroy(plan_1);
    // cufftDestroy(plan_2);
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
