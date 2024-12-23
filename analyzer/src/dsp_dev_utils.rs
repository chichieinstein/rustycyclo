use std::default;

use num::Complex;
use num::{Float, Zero};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

pub fn wgn_complex_f32(n: usize) -> Vec<Complex<f32>> {
    let mut rng = rand::thread_rng();
    let sqrt_2 = f32::sqrt(2.0);
    let normal = Normal::new(0.0, 1.0 / sqrt_2).unwrap();
    let mut wgn_complex: Vec<Complex<f32>> = (0..n)
        .map(|_| Complex::new(normal.sample(&mut rng), normal.sample(&mut rng)))
        .collect();

    wgn_complex
}

pub fn bpsk_symbols(n: usize) -> Vec<Complex<f32>> {
    let mut rng = rand::thread_rng();
    let mut bpsk_symbols: Vec<Complex<f32>> = (0..n)
        .map(|_| {
            let bit = rng.gen_bool(0.5);
            if bit {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(-1.0, 0.0)
            }
        })
        .collect();
    bpsk_symbols
}

pub fn upsample<T>(input: &[T], factor: usize) -> Vec<T>
where
    T: Zero + Clone,
{
    let mut output: Vec<T> = Vec::with_capacity(input.len() * factor);
    for item in input {
        output.push(item.clone());
        output.extend(std::iter::repeat(T::zero()).take(factor - 1));
    }
    output
}

mod dsp_dev_util_tests {

    use super::*;

    #[test]
    fn test_upsample() {
        let input = vec![1, 2, 3];
        let factor = 2;
        let output = upsample(&input, factor);
        assert_eq!(output, vec![1, 0, 2, 0, 3, 0]);
    }
    #[test]
    fn test_wgn_complex_f32() {
        let n = 100000;
        let wgn_complex = wgn_complex_f32(n);
        assert_eq!(wgn_complex.len(), n);

        // find variance
        let mut sum = 0.0;
        for item in wgn_complex.iter() {
            sum += item.re * item.re + item.im * item.im;
        }
        let variance = sum / n as f32;
        println!("Noise variance: {}", variance);
        assert!((variance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bpsk_symbols() {
        let n = 1000000;
        let bpsk_symbols = bpsk_symbols(n);
        assert_eq!(bpsk_symbols.len(), n);

        // check every symbol is either 1 or -1
        for item in bpsk_symbols.iter() {
            assert!(((item.re).abs() - 1.0).abs() < 0.01);
            assert!((item.im - 0.0).abs() < 0.01);
        }

        // check that the sum is zero
        let mut sum = Complex::new(0.0, 0.0);
        for item in bpsk_symbols.iter() {
            sum = sum + item;
        }
        let mean = sum.re / n as f32;
        println!("Mean: {}", mean);
        assert!((mean - 0.0).abs() < 0.01);
    }
}
