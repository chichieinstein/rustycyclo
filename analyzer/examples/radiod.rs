use analyzer::SSCA;
use num::Complex;
use radiod::*;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    const N: i32 = 8192;
    const NP: i32 = 128;
    const SIZE: i32 = 2048;
    let mut k1 = [0; NP];
    let mut e_mat = [0; NP * N];
    let mut ssca = SSCA::new(&mut k1, &mut e_mat, N, NP, SIZE);

    let radio = radiod::devices().unwrap().get_rx().unwrap();
    let mut input = [Complex::<f32>::zero(); 2048];
    let mut output = [0.; N * NP];
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
