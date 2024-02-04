#include <complex.h>
#include <cufft.h>
#include <math.h>
#include <stdio.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../include/spectral_analyzer.cuh"

using std::cout;
using std::endl;
const float pi = acos(-1);
void __global__ make_conj(cufftComplex *in, size_t size) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) in[id] = make_cuComplex(in[id].x, -in[id].y);
}

void __global__ average(cufftComplex *in, float *out, int N, int Np,
			int BATCH) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int id = idy * Np + idx;
	if ((idx < Np) && (idy < N)) {
		for (int idz = 0; idz < BATCH; idz++)
			out[id] += hypotf(in[idz * N * Np + id].x,
					  in[idz * N * Np + id].y);
		out[id] = out[id] / static_cast<float>(BATCH);
	}
}

void __global__ batch_average(cufftComplex *in, float *out, int N, int Np,
			      int BATCH) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	int id2d = idy * Np + idx;

	if ((idx < Np) && (idy < N) && (idz < BATCH)) {
		atomicAdd(&out[id2d],
			  hypotf(in[idz * N * Np + id2d].x / BATCH,
				 in[idz * N * Np + id2d].y / BATCH));
	}
}

void __global__ set_zero(float *in, int max_size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < max_size) {
		in[id] = static_cast<float>(0);
	}
}

void __global__ reductor(float *in, float *out, int N, int Np, int max_size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// int max_size    = 2 * N - (Np / 2);
	if (id < max_size) {
		int reduced_index = (id < N) ? id : id - N + (Np / 2);
		int quotient = (2 * reduced_index) / Np;
		if (id < N) {
			for (int new_ind = 0; new_ind <= quotient; new_ind++) {
				int xind = id - (new_ind * Np) / 2;
				out[id] += in[xind * Np + new_ind];
			}
		} else {
			int counter = (Np - 1) - quotient;
			for (int new_ind = 0; new_ind < counter; new_ind++) {
				int xind = reduced_index + (new_ind * Np) / 2;
				out[id] += in[xind * Np + Np - 1 - new_ind];
			}
		}
	}
}

void __global__ reductor_max(float *in, float *out, int N, int Np,
			     int max_size) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// int max_size    = 2 * N - (Np / 2);
	if (id < max_size) {
		int reduced_index = (id < N) ? id : id - N + (Np / 2);
		int quotient = (2 * reduced_index) / Np;
		if (id < N) {
			for (int new_ind = 0; new_ind <= quotient; new_ind++) {
				int xind = id - (new_ind * Np) / 2;
				out[id] =
				    fmaxf(out[id], in[xind * Np + new_ind]);
			}
		} else {
			int counter = (Np - 1) - quotient;
			for (int new_ind = 0; new_ind < counter; new_ind++) {
				int xind = reduced_index + (new_ind * Np) / 2;
				out[id] = fmaxf(
				    out[id], in[xind * Np + Np - 1 - new_ind]);
			}
		}
	}
}
// Called with stride = true
void __global__ mat_mul_fft_shift_batch_reshape(cufftComplex *window,
						cufftComplex *input,
						cufftComplex *out, int N,
						int Np, int BATCH) {
	int jump{(N + Np)};
	int sign;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;

	int id3d = idz * N * Np + idy * Np + idx;
	int id = idy * Np + idx;

	int input_id = idz * jump + idx + idy;
	int window_id = idx;

	if ((idx < Np) && (idy < N) && (idz < BATCH)) {
		sign = 1 - 2 * (id % 2);
		out[id3d] =
		    make_cuComplex((input[input_id].x * window[window_id].x -
				    input[input_id].y * window[window_id].y) *
				       static_cast<float>(sign),
				   (input[input_id].x * window[window_id].y +
				    input[input_id].y * window[window_id].x) *
				       static_cast<float>(sign));
	}
}

// Called with stride = false
// left_mat should have dimensions N*Np and contains the product of exp_mat and
// kaiser_2
void __global__ mat_vec_multiply_fft_shift_batch_center(
    cufftComplex *left_mat, cufftComplex *input, cufftComplex *right,
    cufftComplex *output_non_conj, cufftComplex *output_conj, int N, int Np,
    int BATCH) {
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

	cufftComplex intermediate_conj;
	cufftComplex intermediate_non_conj;

	if ((idx < Np) && (idy < N) && (idz < BATCH)) {
		int row_pos = (int)id2d / Np;
		sign = 1 - 2 * (row_pos % 2);
		intermediate_non_conj =
		    make_cuComplex(left_mat[id2d].x * input[id_jump].x +
				       left_mat[id2d].y * input[id_jump].y,
				   -left_mat[id2d].x * input[id_jump].y +
				       left_mat[id2d].y * input[id_jump].x);
		intermediate_conj =
		    make_cuComplex(left_mat[id2d].x * input[id_jump].x -
				       left_mat[id2d].y * input[id_jump].y,
				   left_mat[id2d].x * input[id_jump].y +
				       left_mat[id2d].y * input[id_jump].x);
		output_conj[id3d] =
		    make_cuComplex((intermediate_conj.x * right[id3d].x -
				    intermediate_conj.y * right[id3d].y) *
				       static_cast<float>(sign),
				   (intermediate_conj.x * right[id3d].y +
				    intermediate_conj.y * right[id3d].x) *
				       static_cast<float>(sign));
		output_non_conj[id3d] =
		    make_cuComplex((intermediate_non_conj.x * right[id3d].x -
				    intermediate_non_conj.y * right[id3d].y) *
				       static_cast<float>(sign),
				   (intermediate_non_conj.x * right[id3d].y +
				    intermediate_non_conj.y * right[id3d].x) *
				       static_cast<float>(sign));
	}
}

ssca_cuda::ssca_cuda(complex<float> *k1, complex<float> *e_mat, int Nval,
		     int Npval, int size_val, int device) {
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
	n_1 = new int[1];
	n_2 = new int[1];
	*n_1 = Np;
	*n_2 = N;
	inembed_1 = n_1;
	onembed_1 = n_1;
	inembed_2 = n_2;
	onembed_2 = n_2;

	device_id = device;
	cudaSetDevice(device_id);
	cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
	cufftPlanMany(&plan_1, rank, n_1, inembed_1, istride_1, idist_1,
		      onembed_1, ostride_1, odist_1, CUFFT_C2C, batch_1);
	cufftPlanMany(&plan_2, rank, n_2, inembed_2, istride_2, idist_2,
		      onembed_2, ostride_2, odist_2, CUFFT_C2C, batch_2);
	cudaMalloc((void **)&kaiser_1, sizeof(cufftComplex) * Np);
	cudaMalloc((void **)&exp_mat, sizeof(cufftComplex) * N * Np);
	cudaMemcpy(exp_mat, e_mat, sizeof(cufftComplex) * N * Np,
		   cudaMemcpyHostToDevice);
	cudaMemcpy(kaiser_1, k1, sizeof(cufftComplex) * Np,
		   cudaMemcpyHostToDevice);

	cudaMalloc((void **)&inp_buffer, sizeof(cufftComplex) * size);
	cudaMalloc((void **)&inter_gpu, sizeof(cufftComplex) * N * Np * BATCH);
	cudaMalloc((void **)&inter_conj_gpu,
		   sizeof(cufftComplex) * N * Np * BATCH);
	cudaMalloc((void **)&inter_non_conj_gpu,
		   sizeof(cufftComplex) * N * Np * BATCH);

	cudaMalloc((void **)&output_oned_conj_max,
		   sizeof(float) * (2 * N - Np / 2));
	cudaMalloc((void **)&output_oned_conj_sum,
		   sizeof(float) * (2 * N - Np / 2));
	cudaMalloc((void **)&output_oned_non_conj_max,
		   sizeof(float) * (2 * N - Np / 2));
	cudaMalloc((void **)&output_oned_non_conj_sum,
		   sizeof(float) * (2 * N - Np / 2));
	cudaMalloc((void **)&output_non_conj_buffer, sizeof(float) * N * Np);
	cudaMalloc((void **)&output_conj_buffer, sizeof(float) * N * Np);
}

void ssca_cuda::cyclo_gram(cufftComplex *input) {
	dim3 dimBlockMatrixCreation(16, 16, 4);
	dim3 dimGridMatrixCreation(Np / 16, N / 16, BATCH / 4);

	dim3 dimBlockReshape(32, 32);
	dim3 dimGridReshape(Np / 32, N / 32);

	cudaSetDevice(device_id);
	//auto err0 = cudaGetLastError();
	cudaMemcpy(inp_buffer, input, sizeof(cufftComplex) * size,
		   cudaMemcpyHostToDevice);
	mat_mul_fft_shift_batch_reshape<<<dimGridMatrixCreation,
					  dimBlockMatrixCreation>>>(
	    kaiser_1, inp_buffer, inter_gpu, N, Np, BATCH);
	auto arr = cufftExecC2C(plan_1, inter_gpu, inter_gpu, CUFFT_FORWARD);
	//auto err_1 = cudaGetLastError();
	//cout << cudaGetErrorString(err_1) << endl;
	mat_vec_multiply_fft_shift_batch_center<<<dimGridMatrixCreation,
						  dimBlockMatrixCreation>>>(
	    exp_mat, inp_buffer, inter_gpu, inter_non_conj_gpu, inter_conj_gpu,
	    N, Np, BATCH);
	for (int i = 0; i < BATCH; i++) {
		auto err2 =
		    cufftExecC2C(plan_2, inter_conj_gpu + i * N * Np,
				 inter_conj_gpu + i * N * Np, CUFFT_FORWARD);
		auto err1 = cufftExecC2C(
		    plan_2, inter_non_conj_gpu + i * N * Np,
		    inter_non_conj_gpu + i * N * Np, CUFFT_FORWARD);
	//	auto err_3 = cudaGetLastError();
	}
	average<<<dimGridReshape, dimBlockReshape>>>(
	    inter_conj_gpu, output_conj_buffer, N, Np, BATCH);
	//auto err2 = cudaGetLastError();
	//cout << cudaGetErrorString(err2) << endl;
	average<<<dimGridReshape, dimBlockReshape>>>(
	    inter_non_conj_gpu, output_non_conj_buffer, N, Np, BATCH);
	//auto err6 = cudaGetLastError();
	//cout << cudaGetErrorString(err6) << endl;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	//auto err7 = cudaGetLastError();
	//cout << "synchronization error: " << cudaGetErrorString(err7) << endl;
}

void ssca_cuda::reduce_sum() {
	cudaSetDevice(device_id);
	reductor<<<N, 2>>>(output_conj_buffer, output_oned_conj_sum, N, Np,
			   (2 * N - Np / 2));
	reductor<<<N, 2>>>(output_non_conj_buffer, output_oned_non_conj_sum, N,
			   Np, (2 * N - Np / 2));
}

void ssca_cuda::reduce_max() {
	cudaSetDevice(device_id);
	reductor_max<<<N, 2>>>(output_conj_buffer, output_oned_conj_max, N, Np,
			       (2 * N - Np / 2));
	reductor_max<<<N, 2>>>(output_non_conj_buffer, output_oned_non_conj_max,
			       N, Np, (2 * N - Np / 2));
}

void ssca_cuda::dump(float *conj_max, float *conj_sum, float *non_conj_max,
		     float *non_conj_sum) {
	int reductor_size = 2 * N - Np / 2;
	cudaSetDevice(device_id);
	cudaMemcpy(conj_max, output_oned_conj_max,
		   sizeof(float) * reductor_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(non_conj_max, output_oned_non_conj_max,
		   sizeof(float) * reductor_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(conj_sum, output_oned_conj_sum,
		   sizeof(float) * reductor_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(non_conj_sum, output_oned_non_conj_sum,
		   sizeof(float) * reductor_size, cudaMemcpyDeviceToHost);
}

ssca_cuda::~ssca_cuda() {
	cudaSetDevice(device_id);
	cufftDestroy(plan_1);
	cufftDestroy(plan_2);
	delete[] n_1;
	delete[] n_2;

	cudaFree(exp_mat);
	cudaFree(kaiser_1);
	cudaFree(inter_gpu);
	cudaFree(inp_buffer);
	cudaFree(inter_conj_gpu);
	cudaFree(inter_non_conj_gpu);

	cudaFree(output_oned_conj_max);
	cudaFree(output_conj_buffer);
	cudaFree(output_oned_conj_sum);
	cudaFree(output_non_conj_buffer);
	cudaFree(output_oned_non_conj_max);
	cudaFree(output_oned_non_conj_sum);
}
