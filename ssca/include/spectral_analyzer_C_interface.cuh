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
}

#endif