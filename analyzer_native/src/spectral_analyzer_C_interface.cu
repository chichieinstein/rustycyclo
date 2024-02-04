#include <cmath>

#include "../include/spectral_analyzer.cuh"
#include "../include/spectral_analyzer_C_interface.cuh"
using std::cyl_bessel_if;
#include <iostream>
using std::cout;
using std::endl;

extern "C" {
ssca* ssca_create(complex<float>* k1, complex<float>* exp_mat, int Nval,
		  int Npval, int BATCHval, int device) {
	return reinterpret_cast<ssca*>(
	    new ssca_cuda(k1, exp_mat, Nval, Npval, BATCHval, device));
}

void ssca_destroy(ssca* ssca_obj) {
	delete reinterpret_cast<ssca_cuda*>(ssca_obj);
}

void ssca_process(ssca* ssca_obj, complex<float>* input) {
	reinterpret_cast<ssca_cuda*>(ssca_obj)->cyclo_gram(
	    reinterpret_cast<cufftComplex*>(input));
}

void ssca_reduce_sum(ssca* ssca_obj) {
	reinterpret_cast<ssca_cuda*>(ssca_obj)->reduce_sum();
}

void ssca_reduce_max(ssca* ssca_obj) {
	reinterpret_cast<ssca_cuda*>(ssca_obj)->reduce_max();
}

float* allocate_device(int size, int device_id) {
	float* outp;
	cudaSetDevice(device_id);
	cudaMalloc((void**)&outp, sizeof(float) * size);
	return outp;
}

void deallocate_device(float* inp, int device_id) {
	cudaSetDevice(device_id);
	cudaFree(inp);
}

float* allocate_cpu(int size) {
	float* outp;
	outp = new float[size];
	return outp;
}

void deallocate_cpu(float* inp) { delete[] inp; }

void copy_cpu_to_gpu(float* inp, float* outp, int size, int device_id) {
	cudaSetDevice(device_id);
	cudaMemcpy(outp, inp, sizeof(float) * size, cudaMemcpyHostToDevice);
}

void copy_gpu_to_cpu(float* inp, float* outp, int size, int device_id) {
	cudaSetDevice(device_id);
	cudaMemcpy(outp, inp, sizeof(float) * size, cudaMemcpyDeviceToHost);
}

float bessel_func(float inp) { return cyl_bessel_if(0.0, inp); }

void zero_out(ssca* ssca_obj) {
	cudaSetDevice(reinterpret_cast<ssca_cuda*>(ssca_obj)->device_id);
	int Nval = reinterpret_cast<ssca_cuda*>(ssca_obj)->N;
	int Npval = reinterpret_cast<ssca_cuda*>(ssca_obj)->Np;
	int size = 2 * Nval - Npval / 2;
	set_zero<<<size, 1>>>(
	    reinterpret_cast<ssca_cuda*>(ssca_obj)->output_oned_non_conj_sum,
	    size);
	//auto err = cudaGetLastError();
	//cout << "Zero out One: " << cudaGetErrorString(err) << endl;
	set_zero<<<size, 1>>>(
	    reinterpret_cast<ssca_cuda*>(ssca_obj)->output_oned_non_conj_max,
	    size);
	//auto err_2 = cudaGetLastError();
	//cout << "Zero out Two: " << cudaGetErrorString(err_2) << endl;
	set_zero<<<size, 1>>>(
	    reinterpret_cast<ssca_cuda*>(ssca_obj)->output_oned_conj_sum, size);
	//auto err_3 = cudaGetLastError();
	//cout << "Zero out Three: " << cudaGetErrorString(err_3) << endl;
	set_zero<<<size, 1>>>(
	    reinterpret_cast<ssca_cuda*>(ssca_obj)->output_oned_conj_max, size);
	//auto err_4 = cudaGetLastError();
	//cout << "Zero out: " << cudaGetErrorString(err_4) << endl;

}

void ssca_dump(ssca* ssca_obj, float* conj_max, float* conj_sum,
	       float* non_conj_max, float* non_conj_sum) {
	reinterpret_cast<ssca_cuda*>(ssca_obj)->dump(
	    conj_max, conj_sum, non_conj_max, non_conj_sum);
}
}

