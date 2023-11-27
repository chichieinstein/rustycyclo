#include "../include/spectral_analyzer_C_interface.cuh"
#include "../include/spectral_analyzer.cuh"

extern "C"
{
    ssca* ssca_create(complex<float>* k1, complex<float>* k2, complex<float>* exp_mat, int Nval, int Npval, int BATCHval)
    {
        return reinterpret_cast<ssca*>(new ssca_cuda(k1, k2, exp_mat, Nval, Npval, BATCHval));
    }

    void ssca_destroy(ssca* ssca_obj)
    {
        delete reinterpret_cast<ssca_cuda*>(ssca_obj);
    }

    void ssca_process(ssca* ssca_obj, cufftComplex* input, float* output, bool conj, bool uncommented)
    {
        auto ssca_ = reinterpret_cast<ssca_cuda*>(ssca_obj);
        int Nval = ssca_->N;
        int Npval = ssca_->Np;
        int batch = ssca_->BATCH;
        int size = (batch) * ((Nval+Npval) / 2);
        cudaMemcpy(ssca_->in_buffer, input, sizeof(cufftComplex)*size, cudaMemcpyHostToDevice);
        create_batched_center(ssca_->in_buffer, ssca_->center_batch_buffer, Nval, Npval, batch);
        create_batch_matrix_gpu(ssca_->in_buffer, ssca_->ssca_buffer, Nval, Npval, batch);
        ssca_->cyclo_gram(ssca_->ssca_buffer, ssca_->center_batch_buffer, ssca_->ssca_out_buffer, conj, uncommented);
        average<<<Nval, Npval>>>(ssca_->ssca_out_buffer, ssca_->out_gpu, Nval, Npval, batch);
        cudaMemcpy(output, ssca_->out_gpu, sizeof(float)*Nval*Npval, cudaMemcpyDeviceToHost);
    }
}