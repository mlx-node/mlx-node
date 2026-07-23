extern crate napi_build;

fn main() {
    napi_build::setup();
    build_ggml_kquant_reference();
}

/// Compile ggml's own Q4_K / Q5_K / Q6_K decoders, vendored verbatim, as the
/// ground truth for the K-quant parity gate
/// (crates/mlx-core/tests/kquant_ggml_parity.rs).
///
/// `-ffp-contract=off` is load-bearing: `d * sc * q` contracted into an fma
/// would round differently from ggml's own build and quietly move the
/// reference.
///
/// The static library carries three functions no production code path
/// references, so the linker drops it from the cdylib; only the test binaries
/// pull it in.
fn build_ggml_kquant_reference() {
    let dir = "vendor/ggml";
    println!("cargo:rerun-if-changed={dir}/ggml_kquant_ref.c");
    println!("cargo:rerun-if-changed={dir}/ggml_kquant_ref.h");
    cc::Build::new()
        .file(format!("{dir}/ggml_kquant_ref.c"))
        .include(dir)
        .flag("-ffp-contract=off")
        .std("c11")
        .warnings(true)
        .compile("ggml_kquant_ref");
}
