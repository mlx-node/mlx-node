# Bounded Metal residency sets

Port of the MIT-licensed residency implementation in
[mlx.fast revision 8981cef](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine/tree/8981cef5a0a0c5b327f72cf040aa566fc61ff723/Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal).

The pinned MLX backend groups all wired allocations into one Metal residency
set. The reference distributes them across sets capped at 5% of the device's
recommended working set (at least 64 MiB per set). This bounds the amount of
memory affected when the OS changes one set's residency. It does **not** raise
the total wired limit, allocation budget, expert capacity, or system reserves.

The reference's 32-set cap, oversized-allocation handling, allocation accounting,
mutex protection, and queue attachment before every command-buffer commit are
preserved. `MLX_RESIDENCY_SET_MAX_PCT=0` selects one set; the default is `5`.
`MLX_RESIDENCY_DEBUG=1` reports set creation. These are process-start settings.

`build.rs` writes an overlay into `OUT_DIR`, replacing only the Metal device and
residency host implementation. Exact integration anchors fail if the pinned MLX
source changes. CMake and the C++ bridge use the same overlaid headers, and the
install step replaces the corresponding installed headers. The source submodule
and its gitlink remain unchanged. CPU-only and CUDA builds do not use the overlay.

## Validation

The lifecycle and queue-attachment research checks, rollback behavior, and
whole-model results are documented in the
[consolidated report](../../../docs/research/qwen38-flash-next.md).
The standalone harness and raw logs are retained in the local evidence archive.
