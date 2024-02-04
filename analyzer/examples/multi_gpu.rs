use analyzer::SSCAWrapper;
use analyzer::{bpsk_symbols, upsample};

fn main() {
    let size = 133120;
    let mut sscawrapper0 = SSCAWrapper::new(size, None);
    let mut sscawrapper1 = SSCAWrapper::new(size, Some(1));
    let mut sscawrapper2 = SSCAWrapper::new(size, Some(2));
    // get input vector size
    let input_size = sscawrapper0.get_input_size();
    // get output vector size
    let output_size = sscawrapper0.get_output_size();

    let upsample_size = 4;
    let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
    let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

    // get the cycle frequency corresponding to each index of the output vector(s)
    let cycle_vec = sscawrapper0.get_cycles_vec();

    let mut output_conj_max0 = vec![0.0; output_size as usize];
    let mut output_non_conj_sum0 = vec![0.0; output_size as usize];
    let mut output_conj_sum0 = vec![0.0; output_size as usize];
    let mut output_non_conj_max0 = vec![0.0; output_size as usize];

    let mut output_conj_max1 = vec![0.0; output_size as usize];
    let mut output_non_conj_sum1 = vec![0.0; output_size as usize];
    let mut output_conj_sum1 = vec![0.0; output_size as usize];
    let mut output_non_conj_max1 = vec![0.0; output_size as usize];

    let mut output_conj_max2 = vec![0.0; output_size as usize];
    let mut output_non_conj_sum2 = vec![0.0; output_size as usize];
    let mut output_conj_sum2 = vec![0.0; output_size as usize];
    let mut output_non_conj_max2 = vec![0.0; output_size as usize];

    // output_vec_sum contains the sum along the frequency axis
    // output_vec_max contains the max along the frequency axis
    for _ in 0..1000 {
        sscawrapper0.process(
            &mut bpsk_symbols_upsampled,
            &mut output_non_conj_sum0,
            &mut output_non_conj_max0,
            &mut output_conj_sum0,
            &mut output_conj_max0,
        );
        sscawrapper1.process(
            &mut bpsk_symbols_upsampled,
            &mut output_non_conj_sum1,
            &mut output_non_conj_max1,
            &mut output_conj_sum1,
            &mut output_conj_max1,
        );

        sscawrapper2.process(
            &mut bpsk_symbols_upsampled,
            &mut output_non_conj_sum2,
            &mut output_non_conj_max2,
            &mut output_conj_sum2,
            &mut output_conj_max2,
        );
    }

    // Find index where cycle_vec is 0
    let mut cycle_vec_zero_index = 0;
    for (i, cycle) in cycle_vec.iter().enumerate() {
        if (*cycle - 0.0).abs() < 1e-6 {
            cycle_vec_zero_index = i;
            break;
        }
    }

    // Find max value between cycle_vec_zero_index and the end of output_vec_sum
    let mut max_value0 = 0.0;
    let mut max_index0 = 0;
    for (i, value) in output_non_conj_sum0
        .iter()
        .enumerate()
        .skip(cycle_vec_zero_index + 20)
    {
        if *value > max_value0 {
            max_value0 = *value;
            max_index0 = i;
        }
    }

    // Find max value between cycle_vec_zero_index and the end of output_vec_sum
    let mut max_value1 = 0.0;
    let mut max_index1 = 0;
    for (i, value) in output_non_conj_sum1
        .iter()
        .enumerate()
        .skip(cycle_vec_zero_index + 20)
    {
        if *value > max_value1 {
            max_value1 = *value;
            max_index1 = i;
        }
    }

    // Find max value between cycle_vec_zero_index and the end of output_vec_sum
    let mut max_value2 = 0.0;
    let mut max_index2 = 0;
    for (i, value) in output_non_conj_sum2
        .iter()
        .enumerate()
        .skip(cycle_vec_zero_index + 20)
    {
        if *value > max_value2 {
            max_value2 = *value;
            max_index2 = i;
        }
    }
    // print max value and cycle frequency at that point
    println!(
        "Max value from GPU 1: {}, Max value from GPU 2: {}, Max value from GPU 3: {}, cycle frequency: {}",
        max_value0, max_value1, max_value2, cycle_vec[max_index0]
    );
}
