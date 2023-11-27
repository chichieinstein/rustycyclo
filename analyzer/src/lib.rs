extern crate ssca_sys;
use std::any;

use num::Complex;
use ssca_sys::{ssca_create, ssca_destroy, ssca_process, Analyzer};

pub struct SSCA {
    opaque_analyzer: *mut Analyzer,
}

impl SSCA {
    pub fn new(
        k1: &mut [Complex<f32>],
        k2: &mut [Complex<f32>],
        exp_mat: &mut [Complex<f32>],
        n: i32,
        np: i32,
        batch: i32,
    ) -> Self {
        Self {
            opaque_analyzer: unsafe {
                ssca_create(
                    k1.as_mut_ptr(),
                    k2.as_mut_ptr(),
                    exp_mat.as_mut_ptr(),
                    n,
                    np,
                    batch,
                )
            },
        }
    }

    pub fn process(
        &mut self,
        inp: &mut [Complex<f32>],
        outp: &mut [f32],
        conj: bool,
        uncommented: bool,
    ) {
        unsafe {
            ssca_process(
                self.opaque_analyzer,
                inp.as_mut_ptr(),
                outp.as_mut_ptr(),
                conj,
                uncommented,
            )
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn works() {
        let mut k1 = vec![
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
        ];

        let mut k2 = vec![
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
            Complex::new(0.0 as f32, 1.1 as f32),
        ];

        let mut inp = vec![Complex::new(0.0, 0.0 as f32); 1000];
        let mut exp_mat = vec![Complex::new(0.0, 0.0 as f32); 24];
        let mut outp = vec![0.0 as f32; 1000];

        let mut analyzer = SSCA::new(&mut k1, &mut k2, &mut exp_mat, 6, 4, 12);

        analyzer.process(&mut inp, &mut outp, false, false);
    }
}
