# DeepSeek-V4.1 (1017) tiny reference fixture

A deterministic synthetic checkpoint, not a sample of the released model. It has
8 layers (compress ratios `[0,0,2,2,2,1,1,1]`, KV sources `[2,5]`, index sources
`[2,5,7]`, candidate source 5), 12 routed experts (top 6) plus one shared expert,
hyper-connections 4, two engram layers `[1,2]` and a 512-token BPE vocabulary
with the real normalizer chain. Every weight-axis size is distinct, so a swapped
axis fails a shape check.

| File | What |
|---|---|
| `tiny-q8_0.gguf` | The tiny release-form checkpoint (fp8 e4m3 + ue8m0 32x32 blocks, fp4 experts, fp8 engram rows) converted by the llama.cpp PR #30817 converter, then BF16 2-D tensors requantized to Q8_0. Tensor types per kind match the real UD-Q4_K_XL file: dense Q8_0 / F32, experts MXFP4, engram table MXFP8 (type 43). |
| `oracle.npz` | Oracle outputs (DeepSeek 1017 `model.py` on CPU torch, weights = this GGUF dequantized to f32). Keys `<case>.<mode>.<what>`; `oracle.json` → `npz_index` lists dtype and shape. bf16 tensors are stored as `uint16` bit patterns under `<key>@bf16`. |
| `oracle.json` | Cases, modes, per-call logits sha256, dtype per numerics site, adaptations, coverage report (G2.2), GGUF → reference mapping check (G2.3), fake-quant site check (G2.5). |
| `ops.json` | Op-level vectors: MXFP8 / MXFP4 blocks, fp8 / fp4 activation round trips, e2m1 ties, YaRN tables with fp32 and fp64 angles (tiny and real configs), n-gram hash ids, router selection cases, Sinkhorn cases. Floats are given as u32 bit patterns. |

Cases (same 24 ids for `c1`/`c2`/`c3`, decode is teacher-forced):
`c1` prefill 24, `c2` prefill 12 + 12 decode, `c3` prefill 1 + 23 decode,
`pad` prefill 24 of a sequence that starts with the pad token and contains
BOS/EOS and tokens that share a compressed engram id.
Modes: `fp32_on`, `fp32_off`, `bf16_on`, `bf16_off` (activation dtype, and the
1017 activation fake quant at the KV / latent / indexer q, k / `o` sites).

Selection margins: in fp32 every non-tied top-k boundary is at least 10x wider
than the fp32 noise between call splits (`g22_coverage.f32_floor`). Exact ties
are listed in `g22_coverage.f32_exact_ties`; they go to the lowest index. The
bf16-vs-fp32 near-ties are listed per site in `g22_coverage.near_ties`.

The generator, the converter run and the oracle live in the local research
archive, not in this repository. Never regenerate these files or change expected
outputs to make a failing test pass.
