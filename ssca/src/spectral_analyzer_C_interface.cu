#include "../include/spectral_analyzer_C_interface.cuh"
#include "../include/spectral_analyzer.cuh"
#include <cmath>
using std::cyl_bessel_if;

extern "C"
{
    ssca* ssca_create(complex<float>* k1, complex<float>* exp_mat, int Nval, int Npval, int BATCHval)
    {
        return reinterpret_cast<ssca*>(new ssca_cuda(k1, exp_mat, Nval, Npval, BATCHval));
    }

    void ssca_destroy(ssca* ssca_obj)
    {
        delete reinterpret_cast<ssca_cuda*>(ssca_obj);
    }

    void ssca_process(ssca* ssca_obj, complex<float>* input, float* output, bool conj)
    {
        reinterpret_cast<ssca_cuda*>(ssca_obj)->cyclo_gram(reinterpret_cast<cufftComplex*>(input), output, conj);
    }

    void reduce(float* ssca_2d, float* ssca_1d, int N, int Np, int reductor_size)
    {
        reductor<<<N, 2>>>(ssca_2d, ssca_1d, N, Np, reductor_size);
    }

    float* allocate_device(int size)
    {
        float* outp;
        cudaMalloc((void**)&outp, sizeof(float)*size);
        return outp;
    }

    void deallocate_device(float* inp)
    {
        cudaFree(inp);
    }

    float* allocate_cpu(int size)
    {
        float* outp;
        outp = new float [size];
        return outp;
    }

    void deallocate_cpu(float* inp)
    {
        delete [] inp;
    }

    void copy_cpu_gpu(float* inp, float* outp, int size)
    {
        cudaMemcpy(outp, inp, sizeof(float)*size, cudaMemcpyHostToDevice);
    }

    void copy_gpu_cpu(float* inp, float* outp, int size)
    {
        cudaMemcpy(outp, inp, sizeof(float)*size, cudaMemcpyDeviceToHost);
    }

    float bessel_func(float inp)
    {
        return cyl_bessel_if(0.0, inp);
    }
}