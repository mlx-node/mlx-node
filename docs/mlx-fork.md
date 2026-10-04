# MLX fork

`crates/mlx-sys/mlx` is a git submodule of [mlx-node/mlx](https://github.com/mlx-node/mlx)
(`.gitmodules`). The pinned commit is upstream `ml-explore/mlx` main plus a short
patch stack. Everything that does not have to patch MLX lives in mlx-node, on top of
MLX's public C++ API.

```
ml-explore/mlx main  255328713 (2026-10-02)
        │
        │  + 7 patches   fork main
        ▼
mlx-node/mlx         369fec314  ◄── gitlink at crates/mlx-sys/mlx
```

Commit hashes change on every rebase. Name a patch by its subject.

## The 7 patches

| #   | commit      | patch                                                                                                                         | why it cannot live in mlx-node                                                                                                                                                                                              | upstream candidate |
| --- | ----------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 1   | `ceeb6e28a` | `MLX_METAL_FORCE_NAX` CMake option: build the NAX kernels into a metallib whose deployment floor is below macOS 26.2          | MLX's own `kernels/CMakeLists.txt` builds the NAX kernels only when the floor is ≥ 26.2. There is no public hook. `crates/mlx-sys/build.rs` sets the option ON so one artifact keeps a 26.0 floor and still has NAX kernels | no (packaging)     |
| 2   | `732172e4f` | `MLX_METAL_OP_TRACE=1\|2\|3` per-primitive tracer                                                                             | Hooks the loop in `gpu::eval` (`mlx/backend/metal/eval.cpp`). MLX has no per-primitive callback                                                                                                                             | no (diagnostics)   |
| 3   | `2b829d5fa` | `output_shapes` for `Slice`, `Split`, `Pad`, `Depends`, `AsStrided`, `CustomKernel`                                           | Shapeless compile asks every primitive for its output shapes; these threw. They are virtual methods of MLX's own classes, and `CustomKernel` must store its declared shapes. DFlash2's shapeless verify graph needs them    | yes                |
| 4   | `ce5f992cd` | Array retention batched into one completion handler per command-buffer commit (was one ObjC block per primitive)              | Lives inside `CommandEncoder` and `gpu::eval`                                                                                                                                                                               | yes                |
| 5   | `31cc2daed` | Affine/fp `qmv_wide` instantiated for 2..8 vectors per threadgroup, tile cap 8 when N ≥ 2048; `MLX_QMM_SPLITK_MIN_M` override | The instantiation lists are in MLX's prebuilt metallib and the choice is in MLX's `QuantizedMatmul::eval_gpu`. Moving it out means owning MLX's whole affine/fp dispatch. Per-row reduction order is unchanged              | not proposed       |
| 6   | `c23b1113f` | `MLX_METAL_COMMAND_TRACE=1`: one `[metal-command]` JSON line per command buffer, one `[mlx-evaluation]` line per `eval`       | Lives inside `CommandEncoder::commit` (`mlx/backend/metal/device.cpp`) and `eval_impl` (`mlx/transforms.cpp`)                                                                                                               | no (diagnostics)   |
| 7   | `369fec314` | Residency set: copy the `NSError` text before its autorelease pool drains                                                     | Fixes a use-after-free on the path that reports a failed residency-set creation (`mlx/backend/metal/resident.cpp`). It is an MLX bug                                                                                        | yes                |

The bridge also prints `[mlx-compiled]` lines under `MLX_METAL_COMMAND_TRACE=1`
(`crates/mlx-sys/src/mlx_compiled_graph.cpp`). That part is ours, not a patch.

## What lives in mlx-node instead

These replace former fork patches and fork-only MLX APIs. They run as bridge
primitives over Metal libraries we build, from sources in `crates/mlx-sys/src/`:
JIT-built at first use, except the K-quant kernels, which `build.rs` prebuilds
into `paged_attn.metallib`.

| feature                                                                                          | sources                                                                                                                              | Metal library                                                         | pin-bump gate                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------- |
| GGUF K-quants `q3k q4k q5k q6k iq4nl iq4xs iq3s`: matmul, `gather_qmm`, dequantize (CPU + Metal) | `mlx_kquant.{h,cpp}` (ops, validation, CPU reference), `mlx_kquant_metal.cpp` (Metal dispatch), `metal/kquant/{kquant,kquant_nax}.h` | prebuilt in `paged_attn.metallib` (`metal/kquant/*.metal`)            | `kquant_golden_gate`                                              |
| BF16-x × F32-sidecar affine matmul; native `qmv_wide` for gs 32, 8 bits, 2..8 rows, gen ≥ 15     | `mlx_affine_mixed_qmm.{h,cpp}`, `metal/common/affine_qmv_wide_mixed.metal.inc`                                                       | `mlx_node_affine_qmv_wide_mixed_q8g32_nv<N>`                          | `affine_mixed_golden_gate`                                        |
| Segmented verify SDPA (BF16, D=256) and its planners                                             | `mlx_segmented_sdpa.{h,cpp}`, `mlx_segmented_sdpa_plan.h`, `metal/common/sdpa_segmented.metal.inc`                                   | `mlx_node_sdpa_segmented`; stage 2 is MLX's own `sdpa_vector_2pass_2` | `segmented_sdpa_golden_gate`                                      |
| D=256 full-SDPA probes `mlx_metal_d256_full_sdpa_{available,would_use}`                          | `mlx_stream.cpp`: a copy of upstream's `use_fallback` + `has_fused_kernel` (`mlx/backend/metal/scaled_dot_product_attention.cpp`)    | none                                                                  | `qwen35_d256_sdpa` (re-read the upstream predicate on every bump) |
| Test-only hooks (counters, explicit-device entry points)                                         | `mlx_test_kquant.cpp`, `mlx_test_affine_mixed.cpp`, `mlx_test_counters.h`                                                            | -                                                                     | -                                                                 |

These kernels include MLX headers (`utils`, `steel/gemm/gemm`, `quantized_utils`,
`steel/gemm/gemm_nax`, ...) from the pinned tree: `build.rs` compiles the K-quant
`.metal` files against them and generates the source-text preambles the JIT
libraries use. An MLX bump can change these kernels without any change in our files.

The K-quant `.air` files use MLX's own kernel flags (`-fno-fast-math`, no `-O`, no
`-std`; NAX at a 26.2 minimum, built only under MLX's NAX condition), not the
paged-attention flags, and link after the paged `.air` files so the library keeps
the deployment floor's min-OS stamp. There is no JIT fallback: a missing kernel
throws. `kquant_metallib_names` checks every name the dispatcher can build against
the library, both ways.

## Removed, because upstream now has it

| was in mlx-node or the fork                                  | upstream replacement                                                                                                                          |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/mlx-sys/metal-residency/` overlay                    | `09ebe730b` (#4211): same split into residency sets. `MLX_RESIDENCY_SET_MAX_PCT` (default 5) and `MLX_RESIDENCY_DEBUG`, `mlx/utils.h:205-218` |
| custom-kernel hash overlay and `MLX_METAL_HASH_KERNEL_CACHE` | MLX always keys a custom kernel library as `name_<hash(source)>_<options>` (`mlx/backend/metal/custom_kernel.cpp:51`)                         |
| D=256 NAX SDPA fork commits                                  | `714a7efcb` (#3842), `f99e916be` (#4416) and follow-ups                                                                                       |
| GPU busy time (`gpu_total`) on `MLX_METAL_OP_TRACE` lines    | none. `MLX_METAL_COMMAND_TRACE` prints `gpuStart` / `gpuEnd` per command buffer                                                               |

## Bumping MLX

```
fork    1. rebase the 7 patches on upstream main      work branch mlx-node/rebase-<date>
parent  2. move the gitlink, fix API drift, yarn build:native
        3. golden gates, every route                  must be equal, or attributed
        4. dev suites                                 cargo test, vp test
        5. metallib markers + canaries
        6. benches, ABBA against the old addon
fork    7. force-push it to fork main before the parent PR  (CI clones the gitlink)
```

**1. Rebase.** In the fork clone (`origin` = ml-explore/mlx, `me` = mlx-node/mlx):
`git fetch origin me`, `git switch -c mlx-node/rebase-<date> origin/main`, then cherry-pick
the patches in order from `git log --reverse origin/main..me/main`. Fork `main` carries the
patch stack; it does not mirror upstream. Drop a patch only when upstream has the same
change. Check that the standalone CMake build passes with `-DMLX_METAL_FORCE_NAX=ON` and
`CMAKE_OSX_DEPLOYMENT_TARGET=26.0`. Step 7 is
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
and the K-quant kernels `paged_attn.metallib` must contain (`KQUANT_KERNEL_MARKERS`,
`KQUANT_NAX_KERNEL_MARKERS`).
Update them when upstream renames or removes a kernel, then run
`vp test __test__/core/metallib-select.test.ts`. `yarn build:native` runs the same
check on the built file. The GEMM canaries in `crates/mlx-core/src/test_support.rs`
(`half_gemm_untrustworthy`, `f32_gemm_tf32_degraded`, `sorted_gather_mm_untrustworthy`)
print a line when they gate tests off. A canary that starts or stops firing means MLX's
GEMM accuracy on this machine changed.

**6. Benches.** Same machine, cooled, ABBA order, old addon vs new:
`scripts/benchmark-model.ts` (decode and prefill), `docs/research/splash-qwen38/benchmark.ts`
(DFlash2; compare ms per verify cycle when transcripts differ), and
`crates/mlx-core/tests/qwen35_paged_prefill_operator_bench.rs` for D=256 prefill.
The first run after a bump pays a one-time JIT compile for every changed mixed-affine
or segmented SDPA kernel source; the K-quant kernels are prebuilt.
