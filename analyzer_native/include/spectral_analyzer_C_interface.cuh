#ifndef _AANALYZER_
#define _AANALYZER_
#include "spectral_analyzer.cuh"
#include "cufft.h"

struct ssca;
typedef struct ssca ssca;

extern "C" 
{
    ssca* ssca_create(complex<float>*, complex<float>*, int, int, int, int);
    void ssca_destroy(ssca*);
    void ssca_process(ssca*, complex<float>*);
    void ssca_reduce_max(ssca*);
    void ssca_reduce_sum(ssca*);
    void zero_out(ssca*);

    float* allocate_device(int, int);
    void deallocate_device(float*, int);

    float* allocate_cpu(int);
    void deallocate_cpu(float*);

    void copy_cpu_gpu(float*, float*, int, int);
    void copy_gpu_cpu(float*, float*, int, int);
    float bessel_func(float);
}

#endif
