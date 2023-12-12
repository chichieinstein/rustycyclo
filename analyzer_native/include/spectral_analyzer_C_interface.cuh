#ifndef _AANALYZER_
#define _AANALYZER_
#include "spectral_analyzer.cuh"
#include "cufft.h"

struct ssca;
typedef struct ssca ssca;

extern "C" 
{
    ssca* ssca_create(complex<float>*, complex<float>*, int, int, int);
    void ssca_destroy(ssca*);
    void ssca_process(ssca*, complex<float>*, float*, bool);
    void ssca_reduce_max(ssca*, float*);
    void ssca_reduce_sum(ssca*, float*);
    void zero_out(float*, int);

    float* allocate_device(int);
    void deallocate_device(float*);

    float* allocate_cpu(int);
    void deallocate_cpu(float*);

    void copy_cpu_gpu(float*, float*, int size);
    void copy_gpu_cpu(float*, float*, int size);

    float bessel_func(float);
}

#endif