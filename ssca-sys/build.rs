use cc;
use std::path::Path;

fn main() {
    // Define the path to the pre-compiled library
    let precompiled_lib_path =
        Path::new("/home/rsubbaraman/gitrepos/rustycyclo/analyzer_native/libgpu.a");

    // Check if the pre-compiled library exists
    if precompiled_lib_path.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            precompiled_lib_path.parent().unwrap().to_str().unwrap()
        );
        println!("cargo:rustc-link-lib=static=gpu");
    } else {
        // If the pre-compiled library does not exist, compile it
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .cuda(true)
            .cudart("shared")
            .file("../analyzer_native/src/spectral_analyzer_C_interface.cu")
            .file("../analyzer_native/src/spectral_analyzer.cu")
            .flag("-gencode")
            .flag("arch=compute_75, code=sm_80")
            .compile("libgpu.a");
        println!("cargo:rustc-link-search=native=/usr/local/cuda-11.8/lib64");
        println!("cargo:rustc-link-lib=dylib=cufft");
    }

    println!("cargo:rustc-link-search=native=/usr/local/cuda-12.1/lib64");
    println!("cargo:rustc-link-lib=dylib=cufft");
}
