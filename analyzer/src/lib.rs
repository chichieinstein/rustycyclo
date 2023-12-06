extern crate ssca_sys;
use std::any;

use num::Complex;
use ssca_sys::{
    allocate_cpu, allocate_device, bessel_func, copy_cpu_to_gpu, copy_gpu_to_cpu, deallocate_cpu,
    deallocate_device, ssca_create, ssca_destroy, ssca_process, ssca_reduce2D, Analyzer, zero_out
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
        unsafe { zero_out(self.output_oned_buffer.buffer, self.reductor_size);}
        unsafe { ssca_reduce2D(self.opaque_analyzer, self.output_oned_buffer.buffer);}
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
    use std::io::Write;

    use super::*;

    #[test]
    fn works() {
        use std::f32::consts::PI;
        use std::io::Read;
        /// Set things up
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
