/// WASM compatibility shims for NAPI runtime functions.
///
/// On native, napi::bindgen_prelude::spawn_blocking runs a closure on tokio's
/// blocking thread pool. On WASM (wasm32-wasip1-threads), the same function
/// should work via emnapi's thread pool. This module provides a unified import.
pub use napi::bindgen_prelude::spawn_blocking as run_blocking;
