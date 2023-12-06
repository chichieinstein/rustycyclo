extern crate ssca_sys;
use std::any;

use num::Complex;
use ssca_sys::{
    allocate_cpu, allocate_device, bessel_func, copy_cpu_gpu, copy_gpu_cpu, deallocate_cpu,
    deallocate_device, reduce, ssca_create, ssca_destroy, ssca_process, Analyzer,
};

/// This is a GPU CUDA pointer containing 32 bit floating point values.
pub struct DevicePtr {
    buffer: *mut f32,
}

impl DevicePtr {
    /// Allocates space for size number of float32s on the GPU.
    pub fn new(size: i32) -> Self {
        Self {
            buffer: unsafe { allocate_device(size) },
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
    output_buffer: DevicePtr,
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
}

impl Drop for SSCA {
    fn drop(&mut self) {
        unsafe {
            ssca_destroy(self.opaque_analyzer);
        }
    }
}

mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn works() {
        /// Set things up
        let n: i32 = 8192;
        let np = 128;
        let size: i32 = 133120 * 8;

        let n_float = n as f32;
        let np_float = np as f32;

        let kbeta_1 = 80.0;
        let kbeta_2: f32 = 80.0;

        let mut sum_1: f32;
        let mut sum_2: f32;

        let mut k1: Vec<Complex<f32>> = (0..n)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / n_float - 1.0;
                let carg = kbeta_2 * (1.0 - arg * arg).sqrt();
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_1) }, 0.0)
            })
            .collect();

        let mut k2: Vec<Complex<f32>> = (0..np)
            .map(|x| {
                let y = x as f32;
                let arg = 2.0 * y / n_float - 1.0;
                let carg = kbeta_2 * (1.0 - arg * arg).sqrt();
                Complex::new(unsafe { bessel_func(carg) / bessel_func(kbeta_2) }, 0.0)
            })
            .collect();

        sum_1 = k1.iter().fold(0.0, |acc, x| acc + x.re);
        sum_2 = k2.iter().fold(0.0, |acc, x| acc + x.re * x.re);

        sum_2 = sum_2.sqrt();

        k1.iter_mut().for_each(|x| *x = (*x) / sum_1);
        k2.iter_mut().for_each(|x| *x = (*x) / sum_2);

        let mut exp_mat = vec![Complex::new(0.0 as f32, 0.0); (n * np) as usize];

        exp_mat
            .chunks_mut(np as usize)
            .zip((0..n))
            .for_each(|(x, ind0)| {
                for (ind1, item) in x.iter_mut().enumerate() {
                    let exp_arg = -0.5 + (ind0 as f32) / np_float;
                    (*item) = k2[ind1]
                        * Complex::new(
                            (2.0 * PI * exp_arg * (ind0 as f32)).cos(),
                            -(2.0 * PI * exp_arg * (ind0 as f32)).cos(),
                        );
                }
            });
        /// Create SSCA object
        let mut Obj = SSCA::new(&mut k1, &mut exp_mat, n, np, size);

        // // Read data from file
        // let mut file = std::fs::File::open("../busyBand/DSSS.32cf").unwrap();
        // let mut samples_bytes = Vec::new();
        // let _ = file.read_to_end(&mut samples_bytes);
        // let samples: &[f32] = bytemuck::cast_slice(&samples_bytes);

        // // Copy onto input
        // let mut input_vec = vec![0.0 as f32; (nch * nslice) as usize];
        // input_vec[..samples.len()].clone_from_slice(samples);

        // // Setup the output buffer
        // let mut output_buffer: DevicePtr = DevicePtr::new(nch * nslice);

        // // Process
        // chann_obj.process(&mut input_vec, &mut output_buffer);

        // // Display
        // output_buffer.display(100);
    }
}
