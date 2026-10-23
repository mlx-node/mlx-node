//! Run in a fresh process for each MLX_METAL_SERIAL_ENCODER=0/1 setting.
#[test]
#[ignore = "exclusive fresh Metal process; run separately with serial encoder OFF and ON"]
fn encoder_workspace_tracks_reuse_fences_completion_and_teardown() {
    assert!(unsafe { mlx_sys::mlx_test_persistent_workspace_encoder() });
}
