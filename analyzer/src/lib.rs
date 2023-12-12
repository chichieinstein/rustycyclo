extern crate ssca_sys;
use std::any;

use num::Complex;
use ssca_sys::{
    allocate_cpu, allocate_device, bessel_func, copy_cpu_to_gpu, copy_gpu_to_cpu, deallocate_cpu,
    deallocate_device, ssca_create, ssca_destroy, ssca_process, ssca_reduce2D, zero_out, Analyzer,
};

/// This is a GPU CUDA pointer containing 32 bit floating point values.
pub struct DevicePtr {
    buffer: *mut f32,
    size: i32,
}

impl DevicePtr {
    /// Allocates space for size number of float32s on the GPU.
    pub fn new(size: i32) -> Self {
        Self {
            buffer: unsafe { allocate_device(size) },
            size,
        }
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        unsafe { deallocate_device(self.buffer) };
    }
}

pub struct SSCA {
    opaque_analyzer: *mut Analyzer,
    n: i32,
    np: i32,
    size: i32,
    reductor_size: i32,
    output_buffer: DevicePtr,
    output_oned_buffer: DevicePtr,
}

impl SSCA {
    pub fn new(
        k1: &mut [Complex<f32>],
        exp_mat: &mut [Complex<f32>],
        n: i32,
        np: i32,
        size: i32,
    ) -> Self {
        Self {
            opaque_analyzer: unsafe {
                ssca_create(k1.as_mut_ptr(), exp_mat.as_mut_ptr(), n, np, size)
            },
            output_buffer: DevicePtr::new(n * np),
            output_oned_buffer: DevicePtr::new(2 * n - np / 2),
            n,
            np,
            size,
            reductor_size: 2 * n - np / 2,
        }
    }

    pub fn process(&mut self, inp: &mut [Complex<f32>], conj: bool) {
        unsafe {
            ssca_process(
                self.opaque_analyzer,
                inp.as_mut_ptr(),
                self.output_buffer.buffer,
                conj,
            )
        }
    }

    pub fn reduce_feature(&mut self) {
        unsafe {
            zero_out(self.output_oned_buffer.buffer, self.reductor_size);
        }
        unsafe {
            ssca_reduce2D(self.opaque_analyzer, self.output_oned_buffer.buffer);
        }
    }
}

impl Drop for SSCA {
    fn drop(&mut self) {
        unsafe {
            ssca_destroy(self.opaque_analyzer);
        }
    }
}

fn get_cycles_vec(output_size: i32, n: i32, np: i32) -> Vec<f32> {
    let mut cycles_vec = vec![0.0; output_size as usize];

    // Do arange(-0.5,0.5,1/n)
    let mut q_vec = vec![0.0; n as usize];
    for i in 0..n {
        q_vec[i as usize] = -0.5 + (i as f32) / (n as f32);
    }
    let mut k_vec = vec![0.0; np as usize];
    for i in 0..np {
        k_vec[i as usize] = -0.5 + (i as f32) / (np as f32);
    }

    for ind in 0..output_size {
        let reduced_index = if ind < n { ind } else { ind - n + (np / 2) };

        if ind < n {
            cycles_vec[ind as usize] = q_vec[reduced_index as usize] + k_vec[0];
        } else {
            cycles_vec[ind as usize] = q_vec[reduced_index as usize] + k_vec[(np - 1) as usize];
        }
    }

    cycles_vec
}
pub struct SSCAWrapper {
    ssca_handle: SSCA,
    input_size: i32,
    output_size: i32,
    ssca_output_buffer_1d: Vec<f32>,
    cycles_vec: Vec<f32>,
}

impl SSCAWrapper {
    fn new() -> Self {
        use std::f32::consts::PI;

        let n: i32 = 8192;
        let np: i32 = 128;
        let size: i32 = 133120 * 8;

        let n_float = n as f32;
        let np_float = np as f32;

        let kbeta_1 = 80.0;
        let kbeta_2: f32 = 80.0;

        let mut sum_1: f32;
        let sum_2: f32;

        let mut k1: Vec<Complex<f32>> = (0..np)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / np_float - 1.0;
                let carg = kbeta_1 * ((1.0 - arg * arg).sqrt());
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_1) }, 0.0)
            })
            .collect();

        let mut k2: Vec<Complex<f32>> = (0..n)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / n_float - 1.0;
                let carg = kbeta_2 * ((1.0 - arg * arg).sqrt());
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_2) }, 0.0)
            })
            .collect();

        sum_1 = k1.iter().fold(0.0, |acc, x| acc + x.re * x.re);
        sum_2 = k2.iter().fold(0.0, |acc, x| acc + x.re);

        sum_1 = sum_1.sqrt();

        k1.iter_mut().for_each(|x| *x = (*x) / sum_1);
        k2.iter_mut().for_each(|x| *x = (*x) / sum_2);

        let mut exp_mat = vec![Complex::new(0.0 as f32, 0.0); (n * np) as usize];

        exp_mat
            .chunks_mut(np as usize)
            .zip(0..n)
            .for_each(|(x, ind0)| {
                for (ind1, item) in x.iter_mut().enumerate() {
                    let exp_arg = -0.5 + (ind1 as f32) / np_float;
                    (*item) = k2[ind0 as usize]
                        * Complex::new(
                            (2.0 * PI * exp_arg * (ind0 as f32)).cos(),
                            -(2.0 * PI * exp_arg * (ind0 as f32)).sin(),
                        );
                }
            });

        // Initialize SSCA here with the computed values
        let ssca = SSCA::new(&mut k1, &mut exp_mat, n, np, size);

        SSCAWrapper {
            // Populate fields if necessary
            ssca_handle: ssca,
            input_size: size,
            output_size: 2 * n - np / 2,
            ssca_output_buffer_1d: vec![0.0; (2 * n - np / 2) as usize],
            cycles_vec: get_cycles_vec(2 * n - np / 2, n, np),
        }
    }

    pub fn get_input_size(&self) -> i32 {
        self.input_size
    }

    pub fn get_output_size(&self) -> i32 {
        self.output_size
    }

    pub fn get_cycles_vec(&self) -> &[f32] {
        &self.cycles_vec
    }

    pub fn process(&mut self, inp: &mut [Complex<f32>], conj: bool) -> &[f32] {
        self.ssca_handle.process(inp, conj);

        self.ssca_handle.reduce_feature();

        unsafe {
            copy_gpu_to_cpu(
                self.ssca_handle.output_buffer.buffer,
                self.ssca_output_buffer_1d.as_mut_ptr(),
                self.output_size,
            );
        }

        // return pointer to the output buffer
        &self.ssca_output_buffer_1d
    }
}

mod tests {
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use std::io::Write;

    use super::*;

    #[test]
    fn sanity() {
        // create sscawrapper
        let mut sscawrapper = SSCAWrapper::new();

        // get input vector size
        let input_size = sscawrapper.get_input_size();
        // get output vector size
        let output_size = sscawrapper.get_output_size();

        // create input vector full of zeros
        let mut input_vec = vec![Complex::new(0.0, 0.0); input_size as usize];

        // process input vector, and store ouptut in output_vec
        let output_vec = sscawrapper.process(&mut input_vec, true);

        // check if output_vec is of the correct size
        assert_eq!(output_vec.len(), output_size as usize);

        // check every element of output_vec is zero
        assert!(output_vec.iter().all(|&x| x == 0.0));

        // create constant input vector
        let mut input_vec = vec![Complex::new(1.0, 0.0); input_size as usize];

        // process input vector, and store ouptut in output_vec
        let output_vec = sscawrapper.process(&mut input_vec, false);

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut gaussian_noise: Vec<Complex<f32>> = (0..input_size)
            .map(|_| Complex::new(normal.sample(&mut rng), normal.sample(&mut rng)))
            .collect();

        let num_step: usize = 4;
        // create a 4 element window going from 1 and dividing by 2 each time
        let mut window = (0..num_step)
            .map(|x| 2.0_f32.powi(x.try_into().unwrap()))
            .collect::<Vec<f32>>();
        for i in (0..gaussian_noise.len()).step_by(num_step) {
            for j in 1..num_step {
                // set the value to gaussian_noise[i] divided by window
                gaussian_noise[i + j] = gaussian_noise[i] / window[j];
            }
        }

        let output_vec = sscawrapper.process(&mut gaussian_noise, false);

        // get  cycles_vec
        let cycles_vec = sscawrapper.get_cycles_vec();

        // print cycles_vec of 64
        println!("{:?}", &cycles_vec[0..64]);
    }

    #[test]
    fn works() {
        use std::f32::consts::PI;
        use std::io::Read;
        // Set things up
        let n: i32 = 8192;
        let np = 128;
        let size: i32 = 133120 * 8;

        let n_float = n as f32;
        let np_float = np as f32;

        let kbeta_1 = 80.0;
        let kbeta_2: f32 = 80.0;

        let mut sum_1: f32;
        let sum_2: f32;

        let mut k1: Vec<Complex<f32>> = (0..np)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / np_float - 1.0;
                let carg = kbeta_1 * ((1.0 - arg * arg).sqrt());
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_1) }, 0.0)
            })
            .collect();

        let mut k2: Vec<Complex<f32>> = (0..n)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / n_float - 1.0;
                let carg = kbeta_2 * ((1.0 - arg * arg).sqrt());
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_2) }, 0.0)
            })
            .collect();

        sum_1 = k1.iter().fold(0.0, |acc, x| acc + x.re * x.re);
        sum_2 = k2.iter().fold(0.0, |acc, x| acc + x.re);

        sum_1 = sum_1.sqrt();

        k1.iter_mut().for_each(|x| *x = (*x) / sum_1);
        k2.iter_mut().for_each(|x| *x = (*x) / sum_2);

        let mut exp_mat = vec![Complex::new(0.0 as f32, 0.0); (n * np) as usize];

        exp_mat
            .chunks_mut(np as usize)
            .zip(0..n)
            .for_each(|(x, ind0)| {
                for (ind1, item) in x.iter_mut().enumerate() {
                    let exp_arg = -0.5 + (ind1 as f32) / np_float;
                    (*item) = k2[ind0 as usize]
                        * Complex::new(
                            (2.0 * PI * exp_arg * (ind0 as f32)).cos(),
                            -(2.0 * PI * exp_arg * (ind0 as f32)).sin(),
                        );
                }
            });
        // Create SSCA object
        let mut Obj = SSCA::new(&mut k1, &mut exp_mat, n, np, size);

        // Read data from file
        let mut file = std::fs::File::open("../dsss_10dB_1.32cf").unwrap();
        let mut samples_bytes = Vec::new();
        let _ = file.read_to_end(&mut samples_bytes);
        let mut input: &mut [Complex<f32>] = bytemuck::cast_slice_mut(&mut samples_bytes);
        let mut input_vec = vec![Complex::new(0.0 as f32, 0.0); size as usize];

        input_vec[..input.len()].clone_from_slice(input);

        // Process to get conjugate features

        Obj.process(&mut input_vec, false);

        let mut output_non_conj = vec![0.0 as f32; (n * np) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_buffer.buffer,
                output_non_conj.as_mut_ptr(),
                n * np,
            )
        };

        Obj.reduce_feature();

        let mut output_non_conj_1D = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_non_conj_1D.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        // Process to get non-conjugate features
        Obj.process(&mut input_vec, true);

        let mut output_conj = vec![0.0 as f32; (n * np) as usize];

        unsafe { copy_gpu_to_cpu(Obj.output_buffer.buffer, output_conj.as_mut_ptr(), n * np) };

        Obj.reduce_feature();

        let mut output_conj_1D = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_conj_1D.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        let mut file1 = std::fs::File::create("../conj_arr.32f").unwrap();

        let outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut output_conj);

        let _ = file1.write_all(outp_slice);

        let mut file2 = std::fs::File::create("../non_conj_arr.32f").unwrap();

        let outp_slice2: &mut [u8] = bytemuck::cast_slice_mut(&mut output_non_conj);

        let _ = file2.write_all(outp_slice2);

        let mut file3 = std::fs::File::create("../conj_arr_oned.32f").unwrap();

        let outp_slice3: &mut [u8] = bytemuck::cast_slice_mut(&mut output_conj_1D);

        let _ = file3.write_all(outp_slice3);

        let mut file4 = std::fs::File::create("../non_conj_arr_oned.32f").unwrap();

        let outp_slice4: &mut [u8] = bytemuck::cast_slice_mut(&mut output_non_conj_1D);

        let _ = file4.write_all(outp_slice4);
    }
}
