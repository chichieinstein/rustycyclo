extern crate ssca_sys;
use std::any;

use num::Complex;
use ssca_sys::{ssca_create, ssca_destroy, ssca_process, Analyzer, reduce, allocate_cpu, allocate_device, deallocate_cpu, deallocate_device};

pub struct DevicePtr
{
    buffer: *mut f32
}

impl DevicePtr {
    pub fn new(size: i32) -> Self {
        Self { buffer: unsafe{allocate_device(size)}}
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self)
    {
        unsafe{deallocate_device(self.buffer)};
    }
}
pub struct SSCA {
    opaque_analyzer: *mut Analyzer,
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
        }
    }

    pub fn process(
        &mut self,
        inp: &mut [Complex<f32>],
        outp: &mut [f32],
        conj: bool,
    ) {
        unsafe {
            ssca_process(
                self.opaque_analyzer,
                inp.as_mut_ptr(),
                outp.as_mut_ptr(),
                conj,
            )
        }
    }
}

impl Drop for SSCA 
{
    fn drop(&mut self)
    {
        unsafe{
            ssca_destroy(self.opaque_analyzer);
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn works() {
        todo!();
    }
}
