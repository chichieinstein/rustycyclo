use analyzer::SSCAWrapper;
use analyzer::{bpsk_symbols, upsample};

fn main() {
    let size = 133120;
    let mut sscawrapper = SSCAWrapper::new(size, None);
    // get input vector size
    let input_size = sscawrapper.get_input_size();
    // get output vector size
    let output_size = sscawrapper.get_output_size();

    let upsample_size = 4;
    let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
    let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

    // get the cycle frequency corresponding to each index of the output vector(s)
    let cycle_vec = sscawrapper.get_cycles_vec();

    let mut output_conj_max = vec![0.0; output_size as usize];
    let mut output_non_conj_sum = vec![0.0; output_size as usize];
    let mut output_conj_sum = vec![0.0; output_size as usize];
    let mut output_non_conj_max = vec![0.0; output_size as usize];

    // output_vec_sum contains the sum along the frequency axis
    // output_vec_max contains the max along the frequency axis
    sscawrapper.process(
        &mut bpsk_symbols_upsampled,
        &mut output_non_conj_sum,
        &mut output_non_conj_max,
        &mut output_conj_sum,
        &mut output_conj_max,
    );

    // Find index where cycle_vec is 0
    let mut cycle_vec_zero_index = 0;
    for (i, cycle) in cycle_vec.iter().enumerate() {
        if (*cycle - 0.0).abs() < 1e-6 {
            cycle_vec_zero_index = i;
            break;
        }
    }

    // Find max value between cycle_vec_zero_index and the end of output_vec_sum
    let mut max_value = 0.0;
    let mut max_index = 0;
    for (i, value) in output_non_conj_sum
        .iter()
        .enumerate()
        .skip(cycle_vec_zero_index + 20)
    {
        if *value > max_value {
            max_value = *value;
            max_index = i;
        }
    }

    // print max value and cycle frequency at that point
    println!(
        "Max value: {}, cycle frequency: {}",
        max_value, cycle_vec[max_index]
    );
}
