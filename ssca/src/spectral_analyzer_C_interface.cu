#include "../include/spectral_analyzer_C_interface.cuh"
#include "../include/spectral_analyzer.cuh"

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
}