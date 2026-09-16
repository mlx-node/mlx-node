# Qwen4 reference fixtures

These are deterministic synthetic checkpoints, not samples of the released
125B model. Each has fewer than one million parameters. The root uses F32;
`bf16/` uses BF16; `bf16/paged/` has head dimensions accepted by the paged kernels
and includes MTP and three-axis rotary reference outputs.

The oracles were generated from `mlx-vlm/mlx_vlm/models/qwen4_exp/` with seed 104.
The GGUF fixtures represent the same tensors with the GGUF layout transforms
(norm offsets, tiled GDN heads, split indexer and combined PLE table). They are
unquantized to isolate layout correctness. Separate native tests construct packed
quantized banks with known codes and compare kernel outputs.

The development-only generators are retained with the local research archive,
not shipped as runtime tooling. Keep generated weights and oracles together;
do not update expected outputs merely to make a failing test pass. See the
[research report](../../../../../docs/research/qwen38-flash-next.md) for the
archive location and validation scope.
