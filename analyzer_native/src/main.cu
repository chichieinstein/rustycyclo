#include "../include/spectral_analyzer.cuh"
#include <complex.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>
using std::cin;
using std::complex;
using std::cout;
using std::cyl_bessel_if;
using std::endl;
using std::milli;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::steady_clock;
using std::ifstream;
using std::ofstream;

float const PI = acosf(-1.0);

int main()
{
    int N = 8192;
    int Np = 128;
    // int size = 132096*8;
    int size = 133120*8;
    // int batch = 32;
    float kbeta_1 = 80.0;
    float kbeta_2 = 80.0;

    int reductor_size = 2*N - Np / 2;

    complex<float> *win_1 = new complex<float>[Np];
    complex<float> *exp_mat = new complex<float>[N * Np];
    complex<float> *win_2 = new complex<float>[N];

    float sum_1;
    float sum_2;

    for (int j = 0; j < N; j++)
    {
        float darg = static_cast<float>(2 * j) / static_cast<float>(N) - 1.0;
        float carg = kbeta_2 * sqrt(1 - darg * darg);
        float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta_2);
        win_2[j] = complex<float>(earg, 0.0);
        sum_2 += earg;
    }

    for (int j=0; j<N; j++)
    {
        win_2[j] = win_2[j] / sum_2;
    }

    for (int j = 0; j < Np; j++)
    {
        float darg = static_cast<float>(2 * j) / static_cast<float>(Np) - 1.0;
        float carg = kbeta_1 * sqrt(1 - darg * darg);
        float earg = cyl_bessel_if(0.0, carg) / cyl_bessel_if(0.0, kbeta_1);
        win_1[j] = complex<float>(earg, 0.0);
        sum_1 += earg*earg;
    }

    for (int j=0; j<Np; j++)
    {
        win_1[j] = win_1[j] / sqrtf(sum_1);
    }

    for(int j=0; j<N; j++)
    {
        for (int i = 0; i < Np; i++)
        {
            float exp_arg = -0.5 + static_cast<float>(i) / Np;
            exp_mat[j * Np + i] = win_2[j]*complex<float>(cosf(2.0 * PI * exp_arg * j), -sinf(2.0 * PI* exp_arg * j));
        }
    }

    // Correctness Tests
    ifstream my_file;
    my_file.open("../dsss_10dB_1.32cf");
    complex<float> *inp = new complex<float>[size];

    my_file.read((char*) inp, sizeof(float)*2*500000);
    my_file.close();

    auto Obj = ssca_cuda(win_1, exp_mat, N, Np, size);
    float* outp;
    cudaMalloc((void**)&outp, sizeof(float)*N*Np);

    float *outp_conj;
    cudaMalloc((void **)&outp_conj, sizeof(float) * N * Np);

    float *outp_non_conj;
    cudaMalloc((void **)&outp_non_conj, sizeof(float) * N * Np);

    float *oned_output;
    cudaMalloc((void **)&oned_output, sizeof(float) * reductor_size);

    float *oned_conj_output;
    cudaMalloc((void **)&oned_conj_output, sizeof(float) * reductor_size);

    float* outp_conj_cpu = new float [N*Np];
    float* outp_non_conj_cpu = new float [N*Np];

    float* oned_output_cpu = new float [reductor_size];
    float* oned_output_conj_cpu = new float [reductor_size];

    cudaEvent_t stop;
    cudaEventCreate(&stop);
    Obj.cyclo_gram(reinterpret_cast<cufftComplex*>(inp), outp_non_conj, false);
    cudaEventRecord(stop);
    cudaStreamWaitEvent(0, stop, 0);
    Obj.cyclo_gram(reinterpret_cast<cufftComplex*>(inp), outp_conj, true);
    cudaEventRecord(stop);

    cudaStreamWaitEvent(0, stop, 0);
    cudaMemcpy(outp_conj_cpu, outp_conj, sizeof(float)*N*Np, cudaMemcpyDeviceToHost);
    for (int i=0; i< 100; i++)
    {
        cout << outp_conj_cpu[i] << endl;
    }
    cudaEventRecord(stop);
    cudaStreamWaitEvent(0, stop, 0);
    cudaMemcpy(outp_non_conj_cpu, outp_non_conj, sizeof(float)*N*Np, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaStreamWaitEvent(0, stop, 0);

    for(int i=0; i<10; i++)
    {
        cout << outp_conj_cpu[i] << endl;
        cout << outp_non_conj_cpu[i] << endl;
    }

    reductor<<<N, 2>>>(outp_non_conj, oned_output, N, Np, reductor_size);
    cudaEventRecord(stop);
    cudaStreamWaitEvent(0, stop, 0);
    reductor<<<N, 2>>>(outp_conj, oned_conj_output, N, Np, reductor_size);
    cudaEventRecord(stop);
    cudaStreamWaitEvent(0, stop, 0);

    cudaMemcpy(oned_output_cpu, oned_output, sizeof(float)*reductor_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(oned_output_conj_cpu, oned_conj_output, sizeof(float)*reductor_size, cudaMemcpyDeviceToHost);
    
    ofstream new_file;
    new_file.open("../non_conj_arr.32f");
    new_file.write((char*)outp_non_conj_cpu, sizeof(float)*N*Np);
    new_file.close();

    ofstream new_file_conj;
    new_file_conj.open("../conj_arr.32f");
    new_file_conj.write((char*)outp_conj_cpu, sizeof(float)*N*Np);
    new_file_conj.close();

    ofstream new_file_oned;
    new_file.open("../non_conj_arr_oned.32f");
    new_file.write((char*)oned_output_cpu, sizeof(float)*reductor_size);
    new_file.close();

    ofstream new_file_one_conj;
    new_file_conj.open("../conj_arr_oned.32f");
    new_file_conj.write((char*)oned_output_conj_cpu, sizeof(float)*reductor_size);
    new_file_conj.close();
    float elapsed_time = 0.0;
    float elapsed_single_time = 0.0;
    
    cudaEvent_t start_1, stop_1;
    cudaEventCreate(&start_1);
    cudaEventCreate(&stop_1);

    // cudaEventRecord(start_1);

    int ntimes = 100;
    cout << "-----------------Time------------------------" << endl;
    for (int j = 0; j < ntimes; j++)
    {
        // auto start = steady_clock::now();
        cudaEventRecord(start_1, 0);
        Obj.cyclo_gram(reinterpret_cast<cufftComplex *>(inp), outp, true);
        reductor<<<N, 2>>>(outp_non_conj, oned_output, N, Np, reductor_size);
        cudaEventRecord(stop_1, 0);
        cudaEventSynchronize(stop_1);
        cudaEventElapsedTime(&elapsed_single_time, start_1, stop_1);
        cout << elapsed_single_time << endl;
        elapsed_time += elapsed_single_time;
    //     // auto end = steady_clock::now();
    //     // auto elapsed = duration<float, milli>(end - start).count();
    //     // elapsed_time += elapsed;
    }

    cout << "Time taken to process " << size << " samples using SSCA is " << elapsed_time / ntimes << " in milliseconds" << endl;

    // cudaHostUnregister(inp);
    delete[] win_1;
    delete [] win_2;
    delete[] exp_mat;
    delete[] inp;
    cudaFree(outp);
    // delete [] outp;
    cudaFree(outp_conj);
    cudaFree(outp_non_conj);
    delete [] outp_conj_cpu;
    delete [] outp_non_conj_cpu;
    delete [] oned_output_conj_cpu;
    delete [] oned_output_cpu;
    cudaFree(oned_output);
    cudaFree(oned_conj_output);
}