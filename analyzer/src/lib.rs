/*!
# Rust interface to CUDA Strip Spectral Correlation Analyzer (SSCA)

```rust
use analyzer::SSCAWrapper;
use analyzer::{bpsk_symbols, upsample};

fn main(){
    let mut sscawrapper = SSCAWrapper::new();
    // get input vector size
    let input_size = sscawrapper.get_input_size();
    // get output vector size
    let output_size = sscawrapper.get_output_size();

    let upsample_size = 4;
    let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
    let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

    // get the cycle frequency corresponding to each index of the output vector(s)
    let cycle_vec = sscawrapper.get_cycles_vec();

    let mut output_vec_max = vec![0.0; output_size as usize];
    let mut output_vec_sum = vec![0.0; output_size as usize];

    // output_vec_sum contains the sum along the frequency axis
    // output_vec_max contains the max along the frequency axis
    sscawrapper.process(&mut bpsk_symbols_upsampled, false, &mut output_vec_sum, &mut output_vec_max);

}
```

*/

extern crate ssca_sys;
use std::any;

mod dsp_dev_utils;

pub use dsp_dev_utils::*;

use num::Complex;
use ssca_sys::{
    allocate_cpu, allocate_device, bessel_func, copy_cpu_to_gpu, copy_gpu_to_cpu, deallocate_cpu,
    deallocate_device, ssca_create, ssca_destroy, ssca_dump, ssca_process, ssca_reduce_max,
    ssca_reduce_sum, zero_out, Analyzer,
};

/// This is a GPU CUDA pointer containing 32 bit floating point values.
/// This is local to a specific GPU.
pub struct DevicePtr {
    buffer: *mut f32,
    size: i32,
    device_id: i32,
}

impl DevicePtr {
    /// Allocates space for size number of float32s on the GPU.
    pub fn new(size: i32, id: Option<i32>) -> Self {
        let device_id = match id {
            Some(num) => num,
            None => 0,
        };
        Self {
            buffer: unsafe { allocate_device(size, device_id) },
            size,
            device_id,
        }
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        unsafe { deallocate_device(self.buffer, self.device_id) };
    }
}

pub struct SSCA {
    opaque_analyzer: *mut Analyzer,
    n: i32,
    np: i32,
    size: i32,
    reductor_size: i32,
    output_oned_conj_max_buffer: Vec<f32>,
    output_oned_conj_sum_buffer: Vec<f32>,
    output_oned_non_conj_max_buffer: Vec<f32>,
    output_oned_non_conj_sum_buffer: Vec<f32>,
}

impl SSCA {
    pub fn new(
        k1: &mut [Complex<f32>],
        exp_mat: &mut [Complex<f32>],
        n: i32,
        np: i32,
        size: i32,
        id: Option<i32>,
    ) -> Self {
        let device_id = match id {
            Some(x) => x,
            None => 0,
        };
        let reductor_size = (2 * n - np / 2) as usize;
        Self {
            opaque_analyzer: unsafe {
                ssca_create(
                    k1.as_mut_ptr(),
                    exp_mat.as_mut_ptr(),
                    n,
                    np,
                    size,
                    device_id,
                )
            },
            output_oned_conj_max_buffer: vec![0.0; reductor_size],
            output_oned_conj_sum_buffer: vec![0.0; reductor_size],
            output_oned_non_conj_max_buffer: vec![0.0; reductor_size],
            output_oned_non_conj_sum_buffer: vec![0.0; reductor_size],
            n,
            np,
            size,
            reductor_size: reductor_size as i32,
        }
    }

    pub fn process(&mut self, inp: &mut [Complex<f32>]) {
        unsafe { ssca_process(self.opaque_analyzer, inp.as_mut_ptr()) }
    }

    pub fn reduce_feature_max(&mut self) {
        unsafe {
            zero_out(self.opaque_analyzer);
            ssca_reduce_max(self.opaque_analyzer);
        }
    }
    pub fn reduce_feature_sum(&mut self) {
        unsafe {
            zero_out(self.opaque_analyzer);
            ssca_reduce_sum(self.opaque_analyzer);
        }
    }

    pub fn dump_res(&mut self) {
        unsafe {
            ssca_dump(
                self.opaque_analyzer,
                self.output_oned_conj_max_buffer.as_mut_ptr(),
                self.output_oned_conj_sum_buffer.as_mut_ptr(),
                self.output_oned_non_conj_max_buffer.as_mut_ptr(),
                self.output_oned_non_conj_sum_buffer,
            );
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
    cycles_vec: Vec<f32>,
    id: i32,
}

impl SSCAWrapper {
    // Keep size = 2^p * (N + Np) for maximal performance for some p
    pub fn new(size_val: i32, device_id: Option<i32>) -> Self {
        use std::f32::consts::PI;

        let id = match device_id {
            Some(x) => x,
            None => 0,
        };

        let n: i32 = 8192;
        let np: i32 = 128;
        let size: i32 = size_val;

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
        let ssca = SSCA::new(&mut k1, &mut exp_mat, n, np, size, Some(id));

        SSCAWrapper {
            // Populate fields if necessary
            ssca_handle: ssca,
            input_size: size,
            output_size: 2 * n - np / 2,
            cycles_vec: get_cycles_vec(2 * n - np / 2, n, np),
            id,
        }
    }

    pub fn get_input_size(&self) -> i32 {
        self.input_size
    }

    pub fn get_output_size(&self) -> i32 {
        self.output_size
    }

    pub fn get_cycles_vec(&self) -> Vec<f32> {
        get_cycles_vec(
            2 * self.ssca_handle.n - self.ssca_handle.np / 2,
            self.ssca_handle.n,
            self.ssca_handle.np,
        )
    }

    pub fn get_n(&self) -> i32 {
        self.ssca_handle.n
    }

    pub fn process(
        &mut self,
        inp: &mut [Complex<f32>],
        ssca_non_conj_sum_buffer: &mut [f32],
        ssca_non_conj_max_buffer: &mut [f32],
        ssca_conj_max_buffer: &mut [f32],
        ssca_conj_sum_buffer: &mut [f32],
    ) {
        // To get normal SSCA, use conj: False
        self.ssca_handle.process(inp);
        self.ssca_handle.reduce_feature_max();
        self.ssca_handle.reduce_feature_sum();
        self.ssca_handle.dump_res();
        ssca_conj_max_buffer.clone_from_slice(&self.ssca_handle.output_oned_conj_max_buffer);
        ssca_conj_sum_buffer.clone_from_slice(&self.ssca_handle.output_oned_conj_sum_buffer);
        ssca_non_conj_max_buffer
            .clone_from_slice(&self.ssca_handle.output_oned_non_conj_max_buffer);
        ssca_non_conj_sum_buffer
            .clone_from_slice(&self.ssca_handle.output_oned_non_conj_sum_buffer);
    }
}

mod tests {
    use rand::prelude::*;
    use rand_distr::{Distribution, Normal};
    use std::io::Write;

    use dsp_dev_utils::{bpsk_symbols, upsample};

    use super::*;

    #[test]
    fn sanity() {
        let size_val = 133120;
        let mut sscawrapper = SSCAWrapper::new(size_val);

        // get input vector size
        let input_size = sscawrapper.get_input_size();
        // get output vector size
        let output_size = sscawrapper.get_output_size();
        let mut output_vec_max = vec![0.0; output_size as usize];
        let mut output_vec_sum = vec![0.0; output_size as usize];

        // create input vector full of zeros
        let mut input_vec = vec![Complex::new(0.0, 0.0); input_size as usize];

        // process input vector, and store ouptut in output_vec
        sscawrapper.process(
            &mut input_vec,
            false,
            &mut output_vec_sum,
            &mut output_vec_max,
        );

        // check if output_vec is of the correct size
        assert_eq!(output_vec_sum.len(), output_size as usize);

        // check every element of output_vec is zero
        assert!(output_vec_sum.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_bpsk_cycles() {
        // create sscawrapper
        let size_val: i32 = 133120;
        let mut sscawrapper = SSCAWrapper::new(size_val);

        // get input vector size
        let input_size = sscawrapper.get_input_size();
        // get output vector size
        let output_size = sscawrapper.get_output_size();

        // get cycles_vec
        let cycles_vec = sscawrapper.get_cycles_vec();
        let cycle_zero_idx = cycles_vec
            .iter()
            .position(|&x| (x - 0.0).abs() < 1e-6)
            .unwrap();
        println!("cycle_zero_idx: {}", cycle_zero_idx);

        let ssca_n = sscawrapper.get_n();

        let upsample_size = 4;
        let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
        let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

        // print first 12 elements of bpsk_symbols_upsampled
        println!("{:?}", &bpsk_symbols_upsampled[0..12]);

        let mut output_vec_max = vec![0.0; output_size as usize];
        let mut output_vec_sum = vec![0.0; output_size as usize];

        sscawrapper.process(
            &mut bpsk_symbols_upsampled,
            false,
            &mut output_vec_sum,
            &mut output_vec_max,
        );

        // get sum of entire output_vec
        let mut ssca_sum_of_sum: f32 = output_vec_sum.iter().sum();
        let mut ssca_sum_of_max: f32 = output_vec_max.iter().sum();

        let fundamental_cycle_idx = ((ssca_n as f32) / (upsample_size as f32)) as i32;
        // create array with first, second and third fundamental cycles
        let expected_cycles = vec![
            cycle_zero_idx,
            cycle_zero_idx + fundamental_cycle_idx as usize,
            cycle_zero_idx + 2 * fundamental_cycle_idx as usize,
        ];

        let prominence_sum_of_sum = expected_cycles
            .iter()
            .map(|&x| output_vec_sum[x] / ssca_sum_of_sum * (output_vec_sum.len() as f32))
            .collect::<Vec<f32>>();

        let prominence_sum_of_max = expected_cycles
            .iter()
            .map(|&x| output_vec_max[x] / ssca_sum_of_max * (output_vec_max.len() as f32))
            .collect::<Vec<f32>>();
        // print expected_cycle/sum
        println!("expected_cycles/sum_of_sum: {:?}", prominence_sum_of_sum);

        // assert that prominence is all more than 10
        assert!(prominence_sum_of_sum.iter().all(|&x| x > 10.0));

        // print expected_cycle/sum
        println!("expected_cycles/sum_of_max: {:?}", prominence_sum_of_max);

        // assert that sum of max prominence is less than sum of sum
        assert!(
            prominence_sum_of_max.iter().sum::<f32>() < prominence_sum_of_sum.iter().sum::<f32>()
        );
        // assert that sum of max prominence is more than 7
        assert!(prominence_sum_of_max.iter().all(|&x| x > 7.0));
    }

    #[test]
    fn works() {
        use std::f32::consts::PI;
        use std::io::Read;
        // Set things up
        let n: i32 = 8192;
        let np = 128;
        let size: i32 = 133120;

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

        input_vec[..(size as usize)].clone_from_slice(&input[..(size as usize)]);

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

        // Max reduction
        Obj.reduce_feature_max();

        let mut output_non_conj_1D_max = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_non_conj_1D_max.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        // Sum reduction
        Obj.reduce_feature_sum();

        let mut output_non_conj_1D_sum = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_non_conj_1D_sum.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        // Process to get non-conjugate features
        Obj.process(&mut input_vec, true);

        let mut output_conj = vec![0.0 as f32; (n * np) as usize];

        unsafe { copy_gpu_to_cpu(Obj.output_buffer.buffer, output_conj.as_mut_ptr(), n * np) };

        // Max reduction
        Obj.reduce_feature_max();

        let mut output_conj_1D_max = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_conj_1D_max.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        // Sum reduction
        Obj.reduce_feature_sum();

        let mut output_conj_1D_sum = vec![0.0 as f32; (2 * n - np / 2) as usize];

        unsafe {
            copy_gpu_to_cpu(
                Obj.output_oned_buffer.buffer,
                output_conj_1D_sum.as_mut_ptr(),
                2 * n - np / 2,
            );
        }

        let mut file1 = std::fs::File::create("../conj_arr.32f").unwrap();

        let outp_slice: &mut [u8] = bytemuck::cast_slice_mut(&mut output_conj);

        let _ = file1.write_all(outp_slice);

        let mut file2 = std::fs::File::create("../non_conj_arr.32f").unwrap();

        let outp_slice2: &mut [u8] = bytemuck::cast_slice_mut(&mut output_non_conj);

        let _ = file2.write_all(outp_slice2);

        let mut file3 = std::fs::File::create("../conj_arr_oned_max.32f").unwrap();

        let outp_slice3: &mut [u8] = bytemuck::cast_slice_mut(&mut output_conj_1D_max);

        let _ = file3.write_all(outp_slice3);

        let mut file4 = std::fs::File::create("../conj_arr_oned_sum.32f").unwrap();

        let outp_slice4: &mut [u8] = bytemuck::cast_slice_mut(&mut output_conj_1D_sum);

        let _ = file4.write_all(outp_slice4);

        let mut file5 = std::fs::File::create("../non_conj_arr_oned_max.32f").unwrap();

        let outp_slice5: &mut [u8] = bytemuck::cast_slice_mut(&mut output_non_conj_1D_max);

        let _ = file5.write_all(outp_slice5);

        let mut file6 = std::fs::File::create("../non_conj_arr_oned_sum.32f").unwrap();

        let outp_slice6: &mut [u8] = bytemuck::cast_slice_mut(&mut output_non_conj_1D_sum);

        let _ = file6.write_all(outp_slice6);
    }
}
