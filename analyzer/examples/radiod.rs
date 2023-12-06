use analyzer::SSCA;
use num::Complex;
use radiod::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use num::Zero;

use ssca_sys::bessel_func;
use std::f32::consts::PI;


fn main() -> anyhow::Result<()> {
    const N: i32 = 8192;
    const NP: i32 = 128;
    const SIZE: i32 = 133120*8;

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

    // let mut k1 = [0; NP];
    // let mut e_mat = [0; NP * N];
    let mut ssca = SSCA::new(&mut k1, &mut exp_mat, N, NP, SIZE);

    let radio = radiod::devices().unwrap().get_rx().unwrap();
    let mut input = [Complex::<f32>::zero(); 2048];
    let mut output = [0. as f32; (N * NP) as usize];
    let stream = radio.rx_stream(0)?;

    let should_stop: Arc<AtomicBool> = Arc::new(false.into());
    let copy = should_stop.clone();

    ctrlc::set_handler(move || {
        println!("stopping");
        copy.store(true, Ordering::SeqCst);
    })
    .unwrap();

    while !should_stop.load(false) {
        let meta = stream.read(&mut input)?;
        println!("got: {:?}", meta);
        ssca.process(&mut input[..meta.samples()], &mut output, false);
        println!("processed SSCA, output: {:?}", output);
    }

    Ok(())
}
