# MLX source patches

The build applies these patches, in filename order, to a generated copy of
the pinned MLX source under Cargo's `OUT_DIR`. It leaves the submodule checkout
unchanged. MLX itself, bridge headers and generated Metal preambles consume
that same source tree. Patch application fails if its source context changes.

Patch and preparation-helper contents participate in the prebuilt build key.
An automatic prebuilt must match that key; otherwise the build compiles the
patched source. On macOS with Metal, an explicit `MLX_PREBUILT_DIR` must also
record the current clean MLX revision, target triple and build key. An old or
unverifiable artifact is rejected because the encoder patches change its ABI.
Use `MLX_PREBUILT=0` to build the current source.

`0001-metal-indirect-dispatch.patch` adds the encoder operation needed by the
grouped MoE expert grids. Callers register the argument buffer as an input;
the operation preserves MLX's normal barrier and dispatch accounting.

`0002-metal-command-trace-binding.patch` adds command-buffer and stream tags
that associate primitive trace lines with their completed GPU command. It
changes diagnostics only. Completion logging precedes command-buffer reuse;
analyzers must consume records in order rather than group solely by address.

`0003-metal-serial-encoder-experiment.patch` supports
`MLX_METAL_SERIAL_ENCODER=1` for a controlled comparison with Splash's serial
compute dispatch. The default stays concurrent. Resource bookkeeping,
cross-encoder fences and stream ordering remain unchanged. Serial Metal
encoders ignore intrapass memory barriers; see Apple's
[compute encoder documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder).

The preparation helper has standalone tests:

```sh
rustc --edition=2024 --test crates/mlx-sys/build_support/mlx_patches.rs -o /tmp/mlx-patches-tests
/tmp/mlx-patches-tests
```

`0004-metal-sparse-input-budget.patch` adds an explicit storage-unit scheduling
estimate for a sparse read-only binding. It does not change actual resource
tracking, barriers, retention or memory limits. A repeated/shared allocation
binding restores the full first-binding charge. The grouped MoE bridge opts in
only with `MLX_SPLASH_MOE_LOGICAL_BUDGET=1`, using the conservative
`min(experts, physical_rows * top_k)` bound for its nine routed weight/sidecar
arrays; shared weights, activations and scratch retain their ordinary charges.
Full prefill, strided views and offset suballocations retain the existing path.
This changes CommandEncoder's private layout and adds a native symbol, so it
requires a matching source-built MLX library; do not use an older explicit
prebuilt override with it.

`0005-metal-persistent-workspace.patch` adds one bounded (16 slots/8 MiB)
encoder-owned scratch profile. `MLX_SPLASH_MOE_WORKSPACE=1` captures a grouped
MoE request for the measured M8/H2048/I512/E256/top8 staged path. Default is
OFF; concurrent scopes, register experts and diagnostic outputs retain fresh
temporaries. Normal resource bindings/fences remain authoritative; persistent
slots are never temporaries or public outputs. Scratch remains resident until
encoder teardown, not model unload. This is an allocation/lifetime experiment,
not a change to scheduling thresholds or arithmetic. The new method and private
encoder fields require the matching source-built MLX ABI and prebuilt key.
