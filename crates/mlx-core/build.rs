fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.contains("wasm") {
        napi_build::setup();
    }
}
