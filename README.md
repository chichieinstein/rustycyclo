# rustycyclo

Strip Spectral Correlation Analyzer (SSCA) implementation CUDA wrapped inside a Rust interface.

![ssca-sysdiag](./docs/ssca_sysdiag.webp)

## Simple usage example (from docs)

```rust
use analyzer::SSCAWrapper;
use analyzer::{bpsk_symbols, upsample};

fn main(){
    let mut sscawrapper = SSCAWrapper::new();
    // get input vector size
    let input_size = sscawrapper.get_input_size();
    // get output vector size
    let output_size = sscawrapper.get_output_size();

    let upsample_size = 4;
    let bpsk_symbols = bpsk_symbols((input_size / upsample_size).try_into().unwrap());
    let mut bpsk_symbols_upsampled = upsample(&bpsk_symbols, upsample_size.try_into().unwrap());

    // get the cycle frequency corresponding to each index of the output vector(s)
    let cycle_vec = sscawrapper.get_cycles_vec();


    // output_vec_sum contains the sum along the frequency axis
    // output_vec_max contains the max along the frequency axis
    let (output_vec_sum, output_vec_max) =
                sscawrapper.process(&mut bpsk_symbols_upsampled, false);


}
```

## Development environment


Since this repository compiles CUDA code, it needs a specific `nvcc` compiler to build along with `cargo`. The `Dockerfile` in the root of this repository can be used to create a container that can be used to develop and compile the library. Use `docker compose up` to bring up the container.
Then `docker exec -it <container_id> /bin/bash`. `cargo` can be used as normal inside the container.


## Benchmarks

`cargo bench` inside the `analyzer` folder to run the `analyzer` benchmarks. The provided benchmarks runs at `~40 MSamples/sec` on an NVIDIA A10 GPU with a server-class Xeon CPU.

