use analyzer::SSCAWrapper;
use analyzer::{bpsk_symbols, upsample};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn ssca_wrapper_benchmark(c: &mut Criterion) {
    let size_val = 133120;
    let device = Some(0);
    let mut sscawrapper = SSCAWrapper::new(size_val, device);
    // get input vector size
    let input_size = sscawrapper.get_input_size();
    // get output vector size
    let output_size = sscawrapper.get_output_size();

    let upsample_size = 4;
    let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
    let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

    let mut output_conj_sum = vec![0.0; output_size as usize];
    let mut output_conj_max = vec![0.0; output_size as usize];
    let mut output_non_conj_max = vec![0.0; output_size as usize];
    let mut output_non_conj_sum = vec![0.0; output_size as usize];

    // Create a benchmark group
    let mut group = c.benchmark_group("ssca_wrapper_benchmark");
    group.throughput(Throughput::Elements((input_size) as u64));
    group.measurement_time(std::time::Duration::new(20, 0));

    // csearchlight.process(&samples, output);
    group.bench_function("ssca_wrapper", |b| {
        b.iter(|| {
            sscawrapper.process(
                &mut bpsk_symbols_upsampled,
                &mut output_non_conj_sum,
                &mut output_non_conj_max,
                &mut output_conj_sum,
                &mut output_conj_max,
            );
            black_box(&output_non_conj_max);
            black_box(&output_non_conj_sum);
            black_box(&output_conj_sum);
            black_box(&output_conj_max);
        });
    });
}

criterion_group!(benches, ssca_wrapper_benchmark);
criterion_main!(benches);
