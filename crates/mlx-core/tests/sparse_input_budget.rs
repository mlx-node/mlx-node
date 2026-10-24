//! Fresh-process regression for real Metal encoder scheduling accounting.
//! The C++ probe binds bounded buffers without executing kernels. Its trace
//! is independently checked by verify-encoder-trace.py in the research artifact.

#[test]
#[ignore = "requires a fresh GPU process with MLX_MAX_MB_PER_BUFFER=1 and MLX_METAL_COMMAND_TRACE=1"]
fn actual_encoder_sparse_input_budget_restores_bindings() {
    assert_eq!(std::env::var("MLX_MAX_MB_PER_BUFFER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("MLX_METAL_COMMAND_TRACE").as_deref(), Ok("1"));
    assert!(unsafe { mlx_sys::mlx_metal_is_available() });
    assert!(unsafe { mlx_sys::mlx_test_sparse_input_budget_encoder() });
}
