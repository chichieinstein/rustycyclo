import scipy.signal as sig
import numpy as np
# from build.analyzer import ssca_for_python
import matplotlib.pyplot as plt 

# class CycloGram():

    # def __init__(self, N, Np, size, beta_1, beta_2):

    #     self.N          =  N 
    #     self.Np         =  Np
    #     self.size       =  size
    #     self.batch      = 2 * size // (N + Np) - 1
    #     self.oned_size  = 2 * N - Np // 2
    #     win_1           = np.kaiser(Np, beta_1)
    #     norm_1          = (np.sum(np.square(win_1)))**0.5
    #     win_2           = np.kaiser(N,  beta_2)
    #     norm_2          = np.sum(win_2)
    #     self.win_1      =  (win_1 / norm_1).astype('complex64')
    #     self.win_2      =  (win_2 / norm_2).astype('complex64')

    #     out             =    np.zeros((N, Np), dtype='complex128')
    #     exp_args        = np.fft.fftshift(np.fft.fftfreq(Np))
    #     for n in range(N):
    #         out[n] = np.exp(-1j * 2 * np.pi * n * exp_args)
    #     self.exp_mat    = out.astype('complex64').reshape(N*Np)

    #     self.ssca_analyzer = ssca_for_python(size, self.win_1, self.win_2, self.exp_mat, N, Np, self.batch)
        
    #     self.cycles     = np.zeros(self.oned_size, dtype='float32')
    #     self.Q, self.K            = np.fft.fftshift(np.fft.fftfreq(N)), np.fft.fftshift(np.fft.fftfreq(Np))
    #     for ind in range(self.oned_size):
    #         reduced_index           =  ind if (ind < N) else (ind - N + (Np // 2))
    #         self.cycles[ind]        =  (self.Q[reduced_index] + self.K[0]) if (ind < N) else (self.Q[reduced_index] + self.K[-1])
    
    # def analyze(self, inp):
    #     flat_inp           = inp.flatten()
    #     outp               = np.zeros(self.N*self.Np, dtype='float32')
    #     conj_outp          = np.zeros(self.N*self.Np, dtype='float32')
    #     oned_outp          = np.zeros(self.oned_size, dtype='float32')
    #     oned_conj_outp     = np.zeros(self.oned_size, dtype='float32')
    #     self.ssca_analyzer.cyclo_gram(flat_inp, outp, True, True)
    #     self.ssca_analyzer.cyclo_gram(flat_inp, conj_outp, False, True)
    #     self.ssca_analyzer.reductor(outp, oned_outp)
    #     self.ssca_analyzer.reductor(conj_outp, oned_conj_outp)
    #     return outp.reshape((self.N, self.Np)), conj_outp.reshape((self.N, self.Np)), oned_outp, oned_conj_outp
    #     # return outp.reshape((self.N, self.Np)), conj_outp.reshape((self.N, self.Np))

if __name__ == '__main__':

    # inp_arr       =  np.fromfile('/workspaces/containerized_ssca/dsss_10dB_1.32cf', dtype='complex64')
    # Obj           =  CycloGram(8192, 128, 50000, 100.0, 200.0)
    # print(len(inp_arr))
    # output, conj_outp, oned_outp, oned_conj_outp = Obj.analyze(inp_arr)
    N, Np = 8192, 128
    reductor_size = 2*N - Np // 2

    output = np.fromfile("non_conj_arr.32f", dtype='float32')
    output = output.reshape((N, Np))
    conj_outp = np.fromfile("conj_arr.32f", dtype='float32')
    conj_outp = conj_outp.reshape((N,Np))
    
    oned_outp_max = np.fromfile("non_conj_arr_oned_max.32f", dtype='float32')
    oned_conj_outp_max = np.fromfile("conj_arr_oned_max.32f", dtype='float32')
    oned_outp_sum = np.fromfile("non_conj_arr_oned_sum.32f", dtype='float32')
    oned_conj_outp_sum = np.fromfile("conj_arr_oned_sum.32f", dtype='float32')

    print(len(output))
    fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2)

    Q = np.fft.fftshift(np.fft.fftfreq(N))
    K = np.fft.fftshift(np.fft.fftfreq(Np))
    cycles     = np.zeros(reductor_size, dtype='float32')
    for ind in range(reductor_size):
        reduced_index           =  ind if (ind < N) else (ind - N + (Np // 2))
        cycles[ind]        =  (Q[reduced_index] + K[0]) if (ind < N) else (Q[reduced_index] + K[-1])
    ax_1.plot(cycles, oned_conj_outp_sum)
    ax_1.set_title('Conjugate 1D Sum Reduction')

    ax_3.plot(cycles, oned_conj_outp_max)
    ax_3.set_title('Conjugate 1D Max Reduction')

    ax_2.plot(cycles, oned_outp_max)
    ax_2.set_title('Non Conjugate 1D Max reduction')

    ax_4.plot(cycles, oned_outp_sum)
    ax_4.set_title('Non Conjugate 1D Sum reduction')
    
    fig.tight_layout()
    fig.savefig('DSSS_7spc_cpp_trial_6_less_averaging_even_more_beta.png')



    