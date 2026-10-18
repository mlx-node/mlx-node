# MLX fork

`crates/mlx-sys/mlx` is a git submodule of [mlx-node/mlx](https://github.com/mlx-node/mlx)
(`.gitmodules`). The pinned commit is upstream `ml-explore/mlx` main plus a short
patch stack. Everything that does not have to patch MLX lives in mlx-node, on top of
MLX's public C++ API.

```
ml-explore/mlx main  bebd6422f (2026-10-14)
        │
        │  + 6 patches   fork main
        ▼
mlx-node/mlx         e68b9ba77  ◄── gitlink at crates/mlx-sys/mlx
```

Commit hashes change on every rebase. Name a patch by its subject.

## The 6 patches

| #   | commit      | patch                                                                                                                                                        | why it cannot live in mlx-node                                                                                                                                                                                                                                                                                                            | upstream candidate |
| --- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 1   | `44bca05bf` | `output_shapes` for `Slice`, `Split`, `Pad`, `Depends`, `AsStrided`, `CustomKernel`                                                                          | Shapeless compile asks every primitive for its output shapes; these threw. They are virtual methods of MLX's own classes, and `CustomKernel` must store its declared shapes. DFlash2's shapeless verify graph needs them                                                                                                                  | yes                |
| 2   | `7601928c4` | Array retention batched into one completion handler per command-buffer commit (was one ObjC block per primitive)                                             | Lives inside `CommandEncoder` and `gpu::eval`                                                                                                                                                                                                                                                                                             | yes                |
| 3   | `2aed730e8` | Affine/fp `qmv_wide` instantiated for 2..8 vectors per threadgroup, tile cap 8 when N ≥ 2048; `MLX_QMM_SPLITK_MIN_M` override                                | The instantiation lists are in MLX's prebuilt metallib and the choice is in MLX's `QuantizedMatmul::eval_gpu`. Moving it out means owning MLX's whole affine/fp dispatch. Per-row reduction order is unchanged                                                                                                                            | not proposed       |
| 4   | `bcad39304` | `MLX_METAL_COMMAND_TRACE`: 1 `[metal-command]` per command buffer, `[mlx-evaluation]` per eval; 2 + `[metal-op]`; 3 + dtypes                                 | Lives inside `CommandEncoder::commit` (`mlx/backend/metal/device.cpp`), the `gpu::eval` loop (`mlx/backend/metal/eval.cpp`) and `eval_impl` (`mlx/transforms.cpp`). MLX has no per-primitive callback                                                                                                                                     | no (diagnostics)   |
| 5   | `df35e4aa0` | Custom kernel library key computed by `fast::metal_kernel()`: fixed header and source hashed once per kernel function, only the generated signature per call | `CustomKernel::eval_gpu` hashed the whole generated source on every evaluation (`mlx/backend/metal/custom_kernel.cpp`): 7% of the model thread on Qwen3.8 Flash Next decode (~600 custom kernel dispatches per token). The key is built where the source is generated (`mlx/backend/common/metal_kernel.cpp`), stored on the primitive and exposed as `CustomKernel::library_name()` (the DFlash2 selector probe in `mlx_dflash2.cpp` looks the library up by it) | yes                |
| 6   | `e68b9ba77` | Barrier tracking: `prev_*` map each resource to the barrier epoch that recorded it; a barrier or `end_encoding` only bumps the epoch; `next_*` are per-dispatch lists | `CommandEncoder::maybeInsertBarrier` (`mlx/backend/metal/device.cpp`) swapped two `unordered_set`s and cleared one per dispatch; libc++ `clear()` zeroes a bucket array that never shrinks, 7% of the model thread as `__bzero` (still 1.3% with a bucket cap). Nothing is cleared on the dispatch path now; barriers and recorded resources are unchanged | yes                |

The bridge also prints `[mlx-compiled]` lines under `MLX_METAL_COMMAND_TRACE` ≥ 1
(`crates/mlx-sys/src/mlx_compiled_graph.cpp`). That part is ours, not a patch.

## What lives in mlx-node instead

These replace former fork patches and fork-only MLX APIs. They run as bridge
primitives over kernels that `build.rs` prebuilds into `paged_attn.metallib`, from
sources in `crates/mlx-sys/src/`. None of them is JIT-built.

| feature                                                                                                                                                                                 | sources                                                                                                                                                        | Metal library                                                                                               | pin-bump gate                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| GGUF K-quants `q2k q3k q4k q5k q6k iq4nl iq4xs` (and the legacy `iq3s8`) and the grid formats `iq1s iq1m iq2xxs iq2xs iq2s iq3xxs iq3s`: matmul, `gather_qmm`, dequantize (CPU + Metal) | `mlx_kquant.{h,cpp}` (ops, validation, CPU reference), `mlx_kquant_metal.cpp` (Metal dispatch), `metal/kquant/{kquant,kquant_nax,kquant_m8_nax,kquant_grid}.h` | prebuilt in `paged_attn.metallib` (`metal/kquant/*.metal`)                                                  | `kquant_golden_gate`                                              |
| BF16-x × F32-sidecar affine matmul; native `qmv_wide` for gs 32, 8 bits, 2..8 rows, gen ≥ 15                                                                                            | `mlx_affine_mixed_qmm.{h,cpp}`, `metal/affine_mixed/affine_qmv_wide_mixed.metal`                                                                               | prebuilt in `paged_attn.metallib` (`mlx_node_affine_qmv_wide_mixed_q8g32_nv<N>`)                            | `affine_mixed_golden_gate`                                        |
| Segmented verify SDPA (BF16, D=256) and its planners                                                                                                                                    | `mlx_segmented_sdpa.{h,cpp}`, `mlx_segmented_sdpa_plan.h`, `metal/segmented_sdpa/sdpa_segmented.metal`                                                         | prebuilt in `paged_attn.metallib` (`mlx_node_sdpa_segmented_*`); stage 2 is MLX's own `sdpa_vector_2pass_2` | `segmented_sdpa_golden_gate`                                      |
| D=256 full-SDPA probes `mlx_metal_d256_full_sdpa_{available,would_use}`                                                                                                                 | `mlx_stream.cpp`: a copy of upstream's `use_fallback` + `has_fused_kernel` (`mlx/backend/metal/scaled_dot_product_attention.cpp`)                              | none                                                                                                        | `qwen35_d256_sdpa` (re-read the upstream predicate on every bump) |
| Test-only hooks (counters, explicit-device entry points)                                                                                                                                | `mlx_test_kquant.cpp`, `mlx_test_affine_mixed.cpp`, `mlx_test_bridge_metallib.cpp`, `mlx_test_counters.h`                                                      | -                                                                                                           | -                                                                 |

The K-quant kernels include MLX headers (`utils`, `steel/gemm/gemm`,
`quantized_utils`, `steel/gemm/gemm_nax`, ...) from the pinned tree: `build.rs`
compiles the K-quant `.metal` files against them and generates the source-text
preambles that other custom kernels reuse. An MLX bump can change these kernels
without any change in our files. The segmented SDPA and mixed-affine sources
include no MLX headers.

All bridge `.air` files use MLX's own kernel flags (`-fno-fast-math`, no `-O`, no
`-std`), not the paged-attention flags. Every `.air` file compiles at the deployment
target, 26.2 by default: the lowest one at which MLX builds its NAX kernels.
`build.rs` fails a Metal build below it, or with a macOS SDK below 26.2, instead of
letting MLX fall back to `MLX_METAL_NO_NAX`. There is no
JIT fallback: a missing kernel throws `PrebuiltKernelMissing`. Segmented SDPA keeps
it apart from "unsupported here": `mlx_segmented_sdpa_max_query_length` returns -1
(not 0) and the forward returns null, so Qwen3.5 verify errors instead of falling
back to concatenated K/V (`segmented_sdpa_missing_kernels`). The paged and MLX
metallib gates also require the MTLB header's declared size (u64 at offset 16) to
equal the file size, which catches a file cut after its name table. `kquant_metallib_names` and
`bridge_metallib_names` check every name the dispatchers can build against the
library, both ways, and build every pipeline, each segmented SDPA function-constant
specialization included.

## Removed, because upstream now has it

| was in mlx-node or the fork                                   | upstream replacement                                                                                                                                                 |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/mlx-sys/metal-residency/` overlay                     | `09ebe730b` (#4211): same split into residency sets. `MLX_RESIDENCY_SET_MAX_PCT` (default 5) and `MLX_RESIDENCY_DEBUG`, `mlx/utils.h:205-218`                        |
| custom-kernel hash overlay and `MLX_METAL_HASH_KERNEL_CACHE`  | MLX always keys a custom kernel library by its name and source hashes (patch 5 moves the hashing into `fast::metal_kernel()`, `mlx/backend/common/metal_kernel.cpp`) |
| D=256 NAX SDPA fork commits                                   | `714a7efcb` (#3842), `f99e916be` (#4416) and follow-ups                                                                                                              |
| `MLX_METAL_FORCE_NAX` (NAX kernels below a 26.2 floor)        | none needed: the floor is 26.2, where MLX builds the NAX kernels itself (`mlx/backend/metal/kernels/CMakeLists.txt`)                                                 |
| GPU busy time (`gpu_total`) on per-primitive trace lines      | none. `MLX_METAL_COMMAND_TRACE` prints `gpuStart` / `gpuEnd` per command buffer                                                                                      |
| Residency set: copy the `NSError` text before its pool drains | `9503cb7e4` (#4623), same fix                                                                                                                                        |

## Bumping MLX

```
fork    1. rebase the 6 patches on upstream main      work branch mlx-node/rebase-<date>
parent  2. move the gitlink, fix API drift, yarn build:native
        3. golden gates, every route                  must be equal, or attributed
        4. dev suites                                 cargo test, vp test
        5. metallib markers + canaries
        6. prebuilt MLX release + mlx-prebuilt.json   same PR
        7. benches, ABBA against the old addon
fork    8. force-push it to fork main before the parent PR  (CI clones the gitlink)
```

**1. Rebase.** In the fork clone (`origin` = ml-explore/mlx, `me` = mlx-node/mlx):
`git fetch origin me`, `git switch -c mlx-node/rebase-<date> origin/main`, then cherry-pick
the patches in order from `git log --reverse origin/main..me/main`. Fork `main` carries the
patch stack; it does not mirror upstream. Drop a patch only when upstream has the same
change. Check that the standalone CMake build passes with `CMAKE_OSX_DEPLOYMENT_TARGET=26.2`
and builds the NAX kernels (no `NAX kernels require` warning). Step 8 is
`git push --force-with-lease=main:<old me/main> me HEAD:main`.

**2. Parent.** `git -C crates/mlx-sys/mlx checkout <new sha>`, then `yarn build:native`.
Read the upstream diff for every MLX function the bridge calls. A new optional
parameter can compile silently with a wrong argument in its slot (this happened to
`gather_qmm` when `global_scale` was added). Re-read the D=256 predicate in upstream
`scaled_dot_product_attention.cpp` against `mlx_stream.cpp`.

**3. Golden gates.** They are `#[ignore]` and their fixtures belong to an M5 Max
(`applegpu_g17s`); any other machine fails them by design. `MLX_METAL_GPU_ARCH` makes
MLX and our dispatchers route as another GPU class. Fixtures: `crates/mlx-core/tests/golden/`.
CI does not run them: run them locally on an M5 Max for every MLX bump and for every
change to the K-quant, mixed-affine or segmented SDPA kernels.

```sh
# "" = this machine (g17s); g17d / g17g = the other segmented SDPA classes
for a in "" applegpu_g17d applegpu_g17g; do
  MLX_METAL_GPU_ARCH=$a cargo test -p mlx-core --release --test kquant_golden_gate \
    --test affine_mixed_golden_gate --test segmented_sdpa_golden_gate -- --include-ignored
done
# routes with K-quant and mixed-affine fixtures only
for a in applegpu_g17p applegpu_g16s applegpu_g14s applegpu_g14d; do
  MLX_METAL_GPU_ARCH=$a cargo test -p mlx-core --release --test kquant_golden_gate \
    --test affine_mixed_golden_gate -- --include-ignored
done
```

The gates refuse to run when `MLX_SDPA_BLOCKS`, `MLX_QMM_SPLITK_MIN_M` or
`MLX_ENABLE_TF32` is set. Each module doc (`crates/mlx-core/tests/*_golden_gate.rs`)
lists its routes.

**Attributing a mismatch.** Every gate also counts which kernel families ran. A counter
failure means the route changed; a digest failure means the bits changed.

| failing lines                                                               | owner                                 | what to do                                                                                                                                                                            |
| --------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| K-quant (any), segmented kernels / widths / planners, mixed-affine `native` | our kernel or planner                 | Must not move. Find the cause: an MLX header we include as source text, MLX's `sdpa_vector_2pass_2`, or a host helper we call. Fix our side, or stop and ask. Never re-capture        |
| mixed-affine `promoted`                                                     | MLX's own F32 `QuantizedMatmul` graph | Name the upstream commit. Prove the kernel bits are unchanged by routing back (e.g. a temporary gate edit that allows `MLX_QMM_SPLITK_MIN_M`). Measure error against an f64 reference |

Never regenerate a fixture without that attribution. There is no capture mode. A
re-capture is a one-off script that refuses to touch `native` lines, in its own commit
that names the upstream commits and carries the accuracy table (example: `544b3a342`).

**4. Dev suites.** `cargo test -p mlx-core`, `cargo test -p mlx-paged-attn`, `vp test`.
Upstream numeric changes in paths no gate covers show up here. The last bump
moved: MX quantize scale rounding (`02adf7b21`, E8M0 now rounds up), `argmax` NaN rule
(`a124ac096`), qmv batch limits and the qmv fast-path K alignment (`e7838d5e3`,
`8056817bd`). Three more were checked and accepted:

- `b400c6ced`: the Metal NVFP4 `fp_quantize` picks each 16-value scale group by SIMD
  lane, not by grid x. Metal only. `mlx convert` runs every op on the CPU
  (`CpuConvertGuard::enter_cpu`, `crates/mlx-core/src/convert.rs:3692`; GGUF import at
  `crates/mlx-core/src/utils/gguf.rs:4540`), so convert output does not move:
  qwen3.5-0.8b with `--q-mode nvfp4 --q-recipe qwen3_5` gives the same
  `model.safetensors` from the old and new addon (sha256 `dcbf1db0…`). Published NVFP4
  checkpoints are unaffected.
- `56e026d8a`: affine and FP weights dequantize in float32 before the cast to the
  activation type, in the tile loaders of `qmm_t`, `qmm_n`, `gather_qmm` and their NAX
  versions. The vector kernels (`qmv`, `qvm`) already decode in float and do not move.
- `365bd0fba`: `Sigmoid` uses `precise::exp`. The prebuilt metallib is built with
  `-fno-fast-math`, so only a sigmoid inside an `mlx::core::compile` graph changes. The
  hot ones are `mlx_sigmoid_mul_compiled` (`crates/mlx-sys/src/mlx_fused_ops.cpp:14`:
  the Qwen3.5 attention gate and the GDN conv SiLU) and the compiled SwiGLU
  (`crates/mlx-sys/src/mlx_qwen35_common.h:23`: MoE `switch_glu` and the quantized
  `MLPVariant`).

The last two are covered together, not per commit, by the bump's teacher-forced NLL
check (old vs new addon, 3072 tokens, deterministic per binary): Qwen3.8-27B mxfp4
+0.06%, Qwen3.6-35B-A3B MoE mxfp4 −0.08%.

**5. Metallib.** `packages/core/metallib-select.ts` names kernels that a healthy
`mlx.metallib` from this pin must contain (`BASE_KERNEL_MARKERS`, `NAX_KERNEL_MARKERS`),
and the bridge kernels `paged_attn.metallib` must contain (`KQUANT_KERNEL_MARKERS`,
`KQUANT_NAX_KERNEL_MARKERS`, `BRIDGE_KERNEL_MARKERS`). An `mlx.metallib` without the
NAX-only families (`MLX_NAX_ONLY_MARKERS`) fails too: it is an `MLX_METAL_NO_NAX` build.
Update them when upstream renames or removes a kernel, then run
`vp test __test__/core/metallib-select.test.ts`. `yarn build:native` runs the same
check on the built file. The GEMM canaries in `crates/mlx-core/src/test_support.rs`
(`half_gemm_untrustworthy`, `f32_gemm_tf32_degraded`, `sorted_gather_mm_untrustworthy`)
print a line when they gate tests off. A canary that starts or stops firing means MLX's
GEMM accuracy on this machine changed.

**6. Prebuilt MLX.** Pushing the gitlink bump runs `.github/workflows/mlx-prebuilt.yml`
(on any branch of this repo) and publishes release `mlx-prebuilt-<sha12>-<key8>`; paste the
sha256 it prints into `crates/mlx-sys/mlx-prebuilt.json` in the same PR (see below).
Until then every build of the PR compiles MLX from source and says so (`mlx-sys:
building MLX from source (mlx submodule is at <sha> but mlx-prebuilt.json pins <sha>)`).
The workflow needs the commit reachable from the fork (step 8, or push the rebase
branch to the fork first).

**7. Benches.** Same machine, cooled, ABBA order, old addon vs new:
`scripts/benchmark-model.ts` (decode and prefill), `docs/research/splash-qwen38/benchmark.ts`
(DFlash2; compare ms per verify cycle when transcripts differ), and
`crates/mlx-core/tests/qwen35_paged_prefill_operator_bench.rs` for D=256 prefill.
The K-quant, mixed-affine and segmented SDPA kernels are prebuilt, so they add no
JIT compile to the first run after a bump.

## Prebuilt MLX

`crates/mlx-sys/build.rs` links upstream MLX either from its own cmake build of the
submodule (~10 min per cargo profile) or from a **prebuilt artifact** downloaded once per
machine. Only MLX itself is prebuilt: the bridge (`libmlx_ffi.a`), its preambles and
`paged_attn.metallib` stay source-built.

```
artifact (mlx-<triple>.tar.gz)        pin (crates/mlx-sys/mlx-prebuilt.json)
├── lib/libmlx.a                      repo          mlx-node/mlx-node
├── lib/libjaccl.a                    mlx_commit    <full sha of the submodule>
├── lib/mlx.metallib                  release_tag   mlx-prebuilt-<sha12>-<key8>
├── include/{mlx,metal_cpp,jaccl}     assets.<triple>.{file,sha256}
└── manifest.json                     ("" sha256 = never download, build from source)
```

`<key8>` is the first 8 hex of the **build key**: sha256 over the portable cmake defines
(`portable_cmake_defines` in build.rs: everything but tool paths and the SDK root) and
every file under `crates/mlx-sys/cmake/`. A define or include change without a submodule
move therefore names a new release, and a pin left on the old one is refused.

Auto mode (default) uses the artifact only when all of these hold: target
`aarch64-apple-darwin` with Metal, the pin has a non-empty `sha256`, the submodule HEAD
equals `mlx_commit` and is clean, `release_tag` equals the tag this checkout derives
(commit + build key), the download succeeds, the digest matches (a mismatch panics; it is
never silently rebuilt), the artifact's `manifest.json` names the same commit, triple and
tag, and its `deployment_target` equals the build's. Otherwise it builds from source and
prints the reason as a `cargo:warning`. The cache is `$MLX_PREBUILT_CACHE_DIR`, else
`$CARGO_TARGET_DIR/mlx-prebuilt`, else `target/mlx-prebuilt`, unpacked at
`<cache>/<release_tag>/<triple>/` with a `.release-tag` marker; it is shared across cargo
profiles.

| env                            | effect                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `MLX_PREBUILT=0`               | source build, no download                                                                                                                  |
| `MLX_PREBUILT_DIR=<dir>`       | use an unpacked artifact as is (no pin/commit check; needs `lib/libmlx.a`, `include/mlx`)                                                  |
| `MLX_PREBUILT_EXPORT=<dir>`    | source build, then write the artifact set into `<dir>` (what the release workflow does)                                                    |
| `MLX_PREBUILT_PROBE=<file>`    | with `MLX_PREBUILT_EXPORT`: write `{mlx_commit, build_key, release_tag}` to `<file>` and stop before cmake (the workflow's pre-build gate) |
| `MLX_PREBUILT_CACHE_DIR=<dir>` | cache root override                                                                                                                        |
| `GITHUB_TOKEN` / `GH_TOKEN`    | sent as `Authorization: Bearer` on the download (rate limits); anonymous retry if rejected                                                 |

MLX bakes `${MLX_METAL_PATH}/mlx.metallib` into `libmlx.a` as its last-resort load path
(`device.cpp`), and the same variable is where cmake writes and installs the metallib
from. An exported build sets it to `<OUT_DIR>/mlx-prebuilt/<release_tag>/kernels`, so
the string in a prebuilt `libmlx.a` names the producer's machine but ends in
`mlx-prebuilt/<release_tag>/kernels/mlx.metallib`. The bridge additionally bakes
`<OUT_DIR>/paged_attn.metallib` (`MLX_NODE_PAGED_ATTN_METALLIB`), which names the mlx-sys
build and is a last-resort runtime lookup, like MLX's. In prebuilt mode `build.rs` hard-links
(or copies) the prebuilt `mlx.metallib` into that `<OUT_DIR>/lib` next to
`paged_attn.metallib`, so `metallib-select.ts` ships `<OUT_DIR>/lib/mlx.metallib` in both
modes and never re-derives the cache location from the environment. In both modes
`build.rs` also copies `mlx.metallib` next to the cargo test binaries
(`target/<profile>/{,deps/}`): MLX loads a colocated metallib before the baked path, so
the copy must track the libmlx that was linked.

**Bump procedure.** Push the moved gitlink: `.github/workflows/mlx-prebuilt.yml` runs on
that push (any branch; `gh workflow run mlx-prebuilt.yml` also works once it is on main),
builds with `MLX_PREBUILT_EXPORT`, tars the export dir, publishes release
`mlx-prebuilt-<sha12>-<key8>` and prints the sha256 in the job summary. Then set
`mlx_commit`, `release_tag` and `assets.aarch64-apple-darwin.sha256` in
`crates/mlx-sys/mlx-prebuilt.json` in the same PR. Locally the same artifact comes from

```sh
MLX_PREBUILT=0 MLX_PREBUILT_EXPORT=/tmp/mlx-prebuilt-export cargo build --release -p mlx-sys
tar -C /tmp/mlx-prebuilt-export -czf mlx-aarch64-apple-darwin.tar.gz .
shasum -a 256 mlx-aarch64-apple-darwin.tar.gz
```
