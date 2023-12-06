use num::Complex;

#[repr(C)]
pub struct Analyzer {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

extern "C" {
    pub fn ssca_create(
        k1: *mut Complex<f32>,
        // k2: *mut Complex<f32>,
        exp_mat: *mut Complex<f32>,
        n: i32,
        np: i32,
        size: i32,
    ) -> *mut Analyzer;

    pub fn ssca_destroy(inp: *mut Analyzer);

    pub fn ssca_process(
        analyzer: *mut Analyzer,
        inp: *mut Complex<f32>,
        outp: *mut f32,
        conj: bool,
    );

    pub fn reduce(inp: *mut f32, outp: *mut f32, n: i32, np: i32, r_size: i32);

    pub fn allocate_device(size: i32) -> *mut f32;

    pub fn allocate_cpu(size: i32) -> *mut f32;

    pub fn deallocate_device(inp: *mut f32);
    pub fn deallocate_cpu(inp: *mut f32);

    pub fn copy_cpu_gpu(inp: *mut f32, outp: *mut f32, size: i32);
    pub fn copy_gpu_cpu(inp: *mut f32, outp: *mut f32, size: i32);

    pub fn bessel_func(inp: f32) -> f32;
}
