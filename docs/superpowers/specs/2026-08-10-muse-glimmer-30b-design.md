# Muse-Glimmer-30B support — design

**Status:** approved scope, pending spec review
**Checkpoint:** `meta-models/Muse-Glimmer-30B` @ `f84ecc3a0ea984a4c04542a84269e3d065350a6e`
**Local path:** `.cache/models/muse-glimmer-30b` (59.55 GB, 1436 tensors, all BF16)
**Reference:** `transformers/src/transformers/models/muse_glimmer/` (added upstream in `fe95f5423d`)
**Worktree:** `.claude/worktrees/muse-glimmer-30b`, branch `worktree-muse-glimmer-30b`, base `2d1fe60e`

## 1. Scope

| In scope | Out of scope |
| -------- | ------------ |
| Text decoder, bf16, streaming chat | Video (`<\|video\|>`, `patch_temporal`, frame sampling) |
| Image path, end to end | MTP / speculative decoding (no draft heads in the checkpoint) |
| Flat KV, block-paged KV, SSD cold tier | Training / SFT / GRPO |
| `mlx convert` quantization recipe | CUDA / Linux |

"Done" means images work end to end. Video is deliberately deferred; the reference
plumbing for it is understood and recorded in §8 so a later pass is cheap.

## 2. What Muse-Glimmer is

`MuseGlimmerForConditionalGeneration` = windowed vision tower + 3-stage projector +
hybrid text decoder. The upstream `modular_muse_glimmer.py` declares its lineage
explicitly, which is the fastest way to see what mlx-node can reuse:

| Component | Derived from | mlx-node status |
| --------- | ------------ | --------------- |
| `MuseGlimmerTextConfig` | `Gemma2Config` | gemma4 config is close |
| `MuseGlimmerRMSNorm` | `Gemma4RMSNorm` | exists |
| `MuseGlimmerTextCenteredRMSNorm` | `Gemma2RMSNorm` | **no analog** (`(1+w)` centering) |
| `MuseGlimmerTextMLP` / `RotaryEmbedding` | `Gemma2` | exists |
| `MuseGlimmerTextAttention` | **`AfmoeAttention`** | **no analog** (sigmoid output gate) |
| `MuseGlimmerTextDecoderLayer` | `Gemma2DecoderLayer` | 4-norm block, gemma4-like |
| `MuseGlimmerTextNormedEmbedding` | `nn.Embedding` | **no analog** (RMS-normed embedding) |
| `MuseGlimmerVisionConfig/Attention/MLP/EncoderLayer` | `Kimi_K25Vision*` | **no analog** (windowed) |
| `MuseGlimmerVisionPatchEmbedder` | `PaddleOCRVisionEmbeddings` | paddleocr_vl exists |
| `MuseGlimmerVisionRotaryEmbedding` | `Gemma4VisionRotaryEmbedding` | exists, kernel reusable |
| `MuseGlimmerVisionAdapter` | — | new, trivial |
| image processor | `Glm4vImageProcessor` | **no analog** (token-capped smart_resize) |

### 2.1 Text decoder — exact forward

52 layers, `hidden 6656`, `ffn 19968`, `32Q/2KV`, `head_dim 128`, `vocab 202048`,
`tie_word_embeddings false`.

```
h = embed_tokens[ids]                         # [B,T,6656] bf16
h = rmsnorm_scaleless(h, 1e-5)                # embed_norm. NOT gemma's sqrt(hidden) multiply

for i in 0..51:
    is_full = ((51 - i) % 4 == 0)             # full at i in {3,7,...,47,51} -> 13 layers
    theta   = 0 if is_full else 500_000.0     # NoPE on exactly the full layers

    x = centered_rmsnorm(h, W_in[i], 1e-5)                  # norm(x) * (1 + w)
      q = Wq x -> [B,32,T,128]; k = Wk x -> [B,2,T,128]; v = Wv x
      q = rmsnorm_scaleless(q, 1e-5) * 3.87                 # per-head over 128, THEN scale
      k = rmsnorm_scaleless(k, 1e-5)                        # same module, no scale on k
      if theta != 0: q, k = rope_half_split(q, k, theta)    # skipped entirely when NoPE
      a = sdpa(q, k, v, scale = 128**-0.5, mask = causal | sliding(2048))
      a = reshape(a, [B,T,4096])
      a = a * sigmoid(Wgate_attn[i] x)                      # gate from the NORMED x
      a = Wo a
    h = h + centered_rmsnorm(a, W_post_attn[i], 1e-8)       # eps 1e-8

    y = centered_rmsnorm(h, W_pre_ffn[i], 1e-5)
      y = Wdown( silu(Wgate_mlp y) * (Wup y) )
    h = h + centered_rmsnorm(y, W_post_ffn[i], 1e-8)        # eps 1e-8

h      = rmsnorm(h, W_final, 1e-5)                          # norm(x) * w  -- NOT centered
logits = 20.0 * tanh( (h @ W_lm^T) * 0.19611613513818404 / 20.0 )
```

Facts that are easy to get wrong and are load-bearing:

- RoPE layout is **half-split (GPT-NeoX)**, one shared `inv_freq` table for the whole
  model. `layer_rope_theta` is consumed **only as a boolean gate**; all 39 non-zero
  entries are the same `500000.0`.
- There is **no** attention-logit softcapping. Gemma2's `attn_logit_softcapping` is
  deleted in the config (`modular_muse_glimmer.py:606`).
- No attention bias, no sinks, no q/k norm weights (the qk-norm is weightless).
- The KV cache stores **post-norm, post-rope K** and **raw V**.
- Sliding window is `[p-2047, p]` — 2048 visible keys **including self**.
  `RotatingKVCache::new(2048, None)` is correct; do not "compensate" with 2047.
- Because the global layers are NoPE, their KV is **position independent**, which is
  unusually friendly to cross-request prefix reuse.
- `qk_scale_factor 3.87` is **on top of** `1/sqrt(128)`. Net multiplier
  `0.34206290539899237`, cross-checked against the reference converter's
  `43.7840518911 / 128`.

### 2.2 Vision tower — exact forward

50 layers, `hidden 1536`, `16 heads`, `head_dim 96`, `patch 14`, `merge 2`.
Full attention at `i in {3,7,...,47}` plus the last layer `49` (13 full, 37 window).

```
patches = flatten(image)                       # [n_patches, 1176], 1176 = 2*3*14*14
e = Linear(1176 -> 1536)(patches)              # no conv, no bias
e = e + bilinear_resample(pos_table[1024,1536], grid_h, grid_w)
h = ln_pre(e)
h = h[window_index]                            # permute hidden AND position_ids
pos = flip(position_ids)[..., :2] + 1          # (w,h), 1-based
for i in 0..49:
    h = layer_i(h, rope2d(pos), cu_seqlens = window if window_layer else full)
h = h[argsort(window_index)]                   # unpermute BEFORE ln_post
h = ln_post(h)
h = pixel_shuffle_2x2(h)                       # channel-major -> [n/4, 6144]
h = gelu_exact(fc2(gelu_exact(fc1(h))))        # 6144 -> 4096 -> 4096
h = perception_emb_norm(vision_projection(h))  # 4096 -> 6656, then weightless RMSNorm
```

- `window_size = pos_emb_height * patch_size = 448 px` = 32x32 patches.
- `6144` is the projector's **input** width (`merge^2 * 1536`), proven by
  `vision_adapter.fc1.weight = [4096, 6144]`. The `6656` gap is closed by
  `vision_projection = [6656, 4096]`.
- `pixel_shuffle` is **channel-major**: within a merged token,
  `index = channel*4 + (in_h*2 + in_w)`, sub-patch order TL,TR,BL,BR.
- Patch flatten order is `(t, c, ph, pw)` — the **opposite** of Qwen2-VL / GLM-4V's
  `(c, t, h, w)`. Both reference processor files carry a warning comment about this.
- Position-embedding resample is `grid_sample(align_corners=False, padding="zeros")`
  with **unclamped** fractions and per-corner validity masks, summed in fp32.
- gelu is **exact erf**, and the projector applies it twice, including after `fc2`.

### 2.3 Prompt surface — ATEM/Onyx, not harmony

```
<|start|>system<|message|>{content}\n\nReasoning strength: {rs}.{tools}\n\n# Valid recipients: …<|eot|>
<|start|>user<|message|>{content}<|eot|>
<|start|>assistant to=self<|message|>{reasoning}<|eom|>
<|start|>assistant to={recipient}<|message|>{content}<|eot|>
<|start|>tool {name}<|message|><tool_output name="{name}">\n{content}\n</tool_output><|eot|>
generation prompt = "<|start|>assistant"        # bare; model emits " to=…<|message|>"
```

Non-reserved special tokens — exactly 15:

| token | id | | token | id |
| --- | --- | --- | --- | --- |
| `<\|begin_of_text\|>` | 200000 | | `<\|image_start\|>` | 200080 |
| `<\|end_of_text\|>` | 200001 (stop) | | `<\|image_end\|>` | 200081 |
| `<\|eom\|>` | 200007 (**not** a stop) | | `<\|vid_start\|>` | 200082 |
| `<\|eot\|>` | 200008 (stop) | | `<\|vid_end\|>` | 200083 |
| `<\|finetune_right_pad\|>` | 200018 | | `<\|vid_frame_separator\|>` | 200087 |
| `<\|start\|>` | 200022 | | `<\|image\|>` | 200090 (**decoy, unused**) |
| `<\|message\|>` | 200023 | | `<\|video\|>` | 200091 |
| | | | `<\|patch\|>` | 200092 |

This is **not** harmony. There are no channel tokens (`<|channel|>`, `<|constrain|>`,
`<|return|>` are absent from all 2048 added tokens), and identical-looking names carry
different ids than `o200k_harmony`. Never import a harmony id table.
`gemma4/model.rs:7289` hardcodes `"<|channel>thought\n"` — that is precisely the thing
not to clone.

`to=`, role names and the whole `<atem:*>` XML are **ordinary BPE text**. So terminator
detection is by token id; XML detection is by string with hold-back.

The checkpoint ships its own machine-readable parse spec at
`tokenizer_config.json -> response_template`:

```
start_anchor      <|start|>assistant
reasoning_content open "to=self<\|message\|>"  close <|eom|>
content           open "to=user<\|message\|>"  close [<|eot|>, <|eom|>]
tool_calls        open '<atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">'  close </atem:invoke>
                  repeats; params via tag_pattern; value_parser json + allow_non_json
                  transform -> {type: function, function: {name, arguments}}
```

`output_parser.rs` is **driven by this spec**, not hand-rolled regexes.

## 3. Approach

New family directory; reuse what is already generic; leave `gemma4` untouched.

The layer topology resembles gemma4 (interleaved sliding/full, logit softcap, VLM), but
almost none of the arithmetic is shared — six text-level differences plus a completely
different vision tower. Generalising gemma4 to absorb this, or adding config flags to it,
would pay refactor risk on the most heavily tested family in the repo to dedupe a
resemblance that is only skin deep. The genuinely reusable parts
(`crates/mlx-core/src/vision/`) were already factored out by the vision-genericize pass.

## 4. Files

New, under `crates/mlx-core/src/models/muse_glimmer/`:

| File | Contents |
| ---- | -------- |
| `config.rs` | `MuseGlimmerConfig`, text/vision sub-configs, layer-kind + NoPE tables, fail-closed asserts |
| `model.rs` | load, forward (flat + paged), `logits_tail()` |
| `attention.rs` | qk-norm x 3.87, optional RoPE, sigmoid output gate |
| `decoder_layer.rs` | 4-norm block, dual epsilon |
| `mlp.rs` | SwiGLU |
| `layer_cache.rs` | per-layer kind -> `RotatingKVCache` / paged / flat |
| `vision.rs`, `vision_embedder.rs`, `vision_window.rs` | tower, patch embed + zero-pad resample, window index |
| `image_processor.rs` | token-capped smart_resize, LANCZOS, `(t,c,ph,pw)` flatten |
| `output_parser.rs` | `response_template`-driven ATEM parser |
| `persistence.rs`, `sliding_sidecar.rs` | quant load, cold-tier sidecar |

Reused as-is: `vision::{VisionRotaryEmbedding, encoder}`, `RotatingKVCache`,
`PagedKVCacheAdapter`, `BlockAllocator`, `LayerKVPool`, `quant_dispatch`,
`nn::{RMSNorm, Linear, Embedding}`, paddleocr_vl's `cu_seqlens` shape.

**Not** reusable despite appearances: `vision::VisionPositionEmbedding` /
`interpolate.rs:88-106` clamp both index and fraction, so they cannot express the
zero-padded resample; and `vision/encoder.rs:416-420` hardcodes tanh-approximation gelu.

Cross-cutting edits are listed in §6.

## 5. Correctness traps

Ranked by probability x silence. Every one produces fluent-but-wrong output, not an error.

1. **Two RMSNorm conventions.** The 4 per-layer norms are centered `(1+w)`; the final
   `norm` is plain `w`. 208 tensors vs 1. mlx-node's `RMSNorm` adds no `+1`. Bake the
   `+1` at load, in persistence only.

   **Classification must be by tensor name, not by tensor statistics.** Measured directly
   over all 209 norm tensors of the real checkpoint (safetensors header parse + bf16
   decode of the norm tensors only):

   | class | min | max | mean |
   | ----- | --: | --: | ---: |
   | `input_layernorm` | -1.00000 | 3.79688 | 0.30957 |
   | `post_attention_layernorm` | -0.88672 | 3.34375 | 0.03046 |
   | `pre_feedforward_layernorm` | -1.00000 | 2.85938 | 0.05941 |
   | `post_feedforward_layernorm` | -0.80859 | 6.75000 | 0.31077 |
   | final `norm.weight` | -4.93750 | 4.37500 | 0.01688 |

   Both plausible statistical guards are refuted by these numbers. "Centered norms have
   min exactly `-1.0`" holds for only 2 of the 4 classes. "Centered weights cluster near 0,
   plain scales near 1" fails because the final norm's mean is `0.0169`, indistinguishable
   from `post_attention_layernorm`'s `0.0305`.

   What survives is a **one-directional** assert: a centered norm must have
   `min >= -1.0` (otherwise `1+w` flips sign), and the final norm violates it at `-4.9375`.
   So: select by name; assert `min >= -1.0` on every tensor selected for `+1` baking and
   fail closed if not; assert the final norm is **not** in the baked set. That catches a
   name-pattern mistake in either direction without pretending the classes are separable
   by value.

   Related: do not copy `Qwen35Recipe`'s norm-shift list — its `"model.norm.weight"`
   suffix matches `model.language_model.norm.weight`, shifting the one norm that must not
   move, while omitting the two `*_feedforward_layernorm` entries that must.
2. **`output_multiplier` x softcap ordering.** Fixed order: `*1/sqrt(26)`, `/20`, `tanh`,
   `*20`. A missed multiplier is invisible under greedy decoding (argmax is invariant) but
   makes temperature, top-p and logprobs all wrong by ~5x. Route every site through one
   private `logits_tail()`; zero bare softcap calls outside it. The sampler already
   implements softcap (`nn/mod.rs:141-205`) — double-capping is the mirror bug.
3. **`qk_scale_factor` mis-application.** 3.87 is on top of `1/sqrt(128)`. Treating it as
   the whole scale is ~44x off; treating it as gemma's `query_pre_attn_scalar` (used as
   `scalar**-0.5`) is ~7.6x off. Store it as a struct field, never a literal duplicated
   across SDPA sites.
4. **NoPE read as identity rotation.** `layer_rope_theta[i] == 0` means *no rotation
   applied*, not `theta = 1`. Polarity is inverted vs gemma4, which gives global layers
   the long theta.
5. **Vision patch flatten order.** `(t,c,ph,pw)`, not `(c,t,h,w)`. Element count matches
   so nothing errors. Harmless for images (the two temporal halves are identical),
   wrong-by-half for video.
6. **`pixel_shuffle` channel-major, and its order relative to the unpermute.** Sequence is
   layers -> `argsort(window_index)` -> `ln_post` -> `pixel_shuffle`. The obvious
   `reshape(N, 4*1536)` is sub-patch-major and compiles fine.
7. **Zero-pad vs edge-replicate pos-embed resample.** Changes only the outer ring of patch
   position embeddings: fluent output, degraded grounding.
8. **gelu flavour.** Exact erf, twice in the projector, and across all 50 vision layers.
9. **Vision RoPE axis order / 1-based positions.** `flip(-1) + 1` gives `(w,h)` 1-based,
   `freq = [fw,fh,fw,fh]`. Assert `inv_freq.len()*4 == head_dim` so a config change fails
   loudly.
10. **Stop-token set — and it is NOT wired through the tokenizer.** Three sources disagree
    and only `generation_config.json` is right: it lists `[200001, 200008]`, while
    `config.json text_config.eos_token_id` is `200001` alone and `tokenizer_config.json
    eos_token` is `<|end_of_text|>`. Nothing on the decode path reads
    `resolve_special_tokens` / `tokenizer.get_eos_token_id()`; the stop site is
    `engine/decode.rs`'s `stops_at_eos = token_id == eos_id ||
    extra_eos_ids.contains(&token_id)`, resolved once per turn from two `ChatBackend`
    hooks. M1 contract: `session_eos_id` (`backend.rs:675`) returns `<|eot|>` 200008;
    `extra_eos_ids` (`backend.rs:947`) returns `gen_defaults.eos_token_ids`, populated at
    load by `parse_generation_defaults` (`persistence.rs:947`). Both overrides are
    mandatory: omitting `extra_eos_ids` **silently** drops `200001` (the trait default is
    `Vec::new()`, and `[].contains()` is always false, so turns stop only on `<|eot|>`),
    and inheriting the ChatML `session_eos_id` (`tok.im_end_id()`) **hard-errors every
    turn** rather than resolving the wrong id, because this checkpoint has no
    `<|im_end|>` — copy gemma4's `turn_end_id()` override (`gemma4/model.rs:7314`).
    `<|eom|>` 200007 must stay OUT of the set: it ends a message, not a turn.
11. **Quantization reaches the cross-modal bridge.** `convert.rs` has zero occurrences of
    `vision_adapter` / `vision_projection` and neither matches any skip substring, so the
    most sensitive tensors in a VLM get packed for 0.139 GB. Also
    `model.language_model.norm.weight` passes the `_norm.` guard and survives only via the
    `ndim<2` rejection.
12. **Placeholder-count mismatch.** HF zeroes placeholder ids before `embed_tokens` and
    asserts `count(placeholders) == features.shape[0]`. Port that equality check.
    The wrapper tokens `200080/200081/200082/200083/200087` must **not** be zeroed — they
    carry real learned embeddings.

## 6. Cross-cutting edits

Blocking / build-breaking:

- `crates/mlx-core/src/tokenizer.rs:1208-1240` — register `items` on `ValueKind::Map`
  (insertion-ordered `[k,v]` pairs). The template uses the **method** form
  `args.items()`; the repo only supports the filter form `args|items`, and the
  `ValueKind::Map` arm handles only `get`, so every render of an assistant message
  carrying `tool_calls` hard-fails with `UnknownMethod`. **This fires on turn 2 of a tool
  loop, not turn 1**, because the tool-definitions block uses no `.items()`. Any test
  matrix must include a full tool round trip.
- `packages/.../__test__/models/model-loader-registry.test.ts:8-19` — `as const satisfies
  Record<ModelType, readonly string[]>`; a new `ModelType` fails `yarn typecheck` /
  `build:ts` / CI lint until the key is added.

Silent if missed:

- `LAUNCH_PRESETS` (`presets.ts:103-124`) — absence makes `discoverModels` drop the
  checkpoint (`host/discover.ts:54-58`).
- `FAMILY_TRAITS` (`models.ts:186-190`) — absence makes the agent skip the family.
- `supportsImages` (`models.ts:132-136`) hardcodes gemma4/qwen3_5/qwen3_5_moe, so a new
  VLM advertises text-only to the model picker.
- `COLD_RESTORE_FAMILIES` (`cold_tier.rs`) **and** `COLD_TIER_RESTORE_FAMILIES`
  (`model-host.ts`), with the drift-guard test `cold-tier-families.test.ts`.
- `tokenizer.rs` `context!` block — add a **pinned** `current_date`. Unpinned, every
  prefix and cold-tier entry invalidates at local midnight, and prompts diverge from any
  HF-rendered fixture.
- `sanitize_chatml_content` (`tokenizer.rs:1035-1040`) strips only ChatML markers, so user
  content containing `<|eot|>` or `<|start|>assistant to=user<|message|>` encodes to real
  ids — role forgery and turn termination from user text. Needs a family-aware deny list.
- `tojson` (`tokenizer.rs:1119-1121`) is `serde_json::to_string` (compact); HF uses
  `json.dumps` default separators `": "` / `", "`. Every tool-enabled system prefix and
  every container-valued ATEM argument is off-distribution by whitespace, and byte-
  mismatches any HF fixture.
- Media ordering: default `MultimodalContentOrder::TextThenMedia` (`tokenizer.rs:1547`)
  puts `<|patch|>` **after** the prose. `render_content` has no `else` branch, so audio
  parts are silently dropped — the `chat_napi` audio guard must be explicitly false.
- Expanded token count is **N+2** per image (`<|image_start|>` + N x `<|patch|>` +
  `<|image_end|>`), so worst case is 4098 prompt positions per image. Write the
  image-admission knob against that number.
- Two committed `index.d.cts` artifacts; only `packages/core` is regenerated.

Robustness, from the reference template's own failure modes:

- Assistant prose is **dropped** when `tool_calls` is present (the branch is
  `if(tool_calls) … else(content)`), so the re-rendered prefix differs from what the model
  emitted — this both loses context and breaks prefix reuse.
- `arguments` that parse to a non-object (`""`, `"[]"`, `"5"`, `"null"`) hit the
  template's `raise_exception`. Pre-check fail-closed before render.
- `<|eom|>` is not a stop, so the model can emit `<|start|>user<|message|>…` and self-play
  a conversation. Needs a role guard plus per-turn message/token caps.
- Validate `to=` and `<atem:invoke name>` against `{self, user} ∪ exact tool names for this
  turn`; the advertised `# Valid recipients:` globs are prose, not the executable set.
- Stream emitter needs a `>= 24`-char hold-back (longest literals are 22 chars) or
  `<atem` fragments leak into content deltas; and the model's first emitted characters are
  ` to=<recipient><|message|>`, which must be fully buffered before routing or the
  recipient text surfaces as content.
- A `developer`-role message renders to nothing (the template branches only on
  system/user/tool/assistant).

## 7. Verification

59.5 GB will not fit a macos-26 runner — the documented qwen3_5-MoE position
(`ci.yml:248-253`). So every real-weights gate stays a local `#[ignore]`, and **all
arithmetic is pinned by model-free unit tests** that do run in CI.

Tier 1 — model-free, in CI:

- both norm conventions, on hand-built tensors: that `(1+w)` is applied to exactly the 4
  per-layer classes and never to the final norm, selected **by name**; plus the
  one-directional `min >= -1.0` assert on the baked set (§5.1)
- `logits_tail()` ordering, incl. a test that fails if the multiplier moves after `tanh`
- `qk_scale`: assert net multiplier `0.34206290539899237`
- NoPE layer map == `{3,7,…,51}`; assert 13 zeros / 39 x `500000.0`
- sliding bound: query at `p` sees exactly 2048 keys including self
- `pixel_shuffle` channel-major, on a labelled tensor where sub-patch-major differs
- patch flatten `(t,c,ph,pw)` with distinguishable temporal halves
- `smart_resize` grid for a table of sizes, with a **pinned tie-break** (HF's `min()` over
  a `set()` is iteration-order dependent, so accept one-grid divergence on exact ties)
- prompt golden strings: no-system default preamble, system-in-place, `<|eom|>` vs
  `<|eot|>` rule, tool defs, **full tool round trip** (catches `.items()`), tool result
  with and without `name`, image part placement
- ATEM parse via `response_template`, incl. malformed input and type round-trip loss

Tier 2 — real weights, local `#[ignore]`:

- logit parity vs an HF reference dump (fixture generated once, committed if small)
- image parity on a fixed image
- `muse_glimmer_paged_vs_flat_parity` — byte-equal greedy, fresh and delta
- cold-tier restore parity, single-gate binaries only

Each milestone's gate must name the mutation it would catch. A green suite that would stay
green under an injected `+1` on the final norm is not a gate.

## 8. Milestones

| # | Deliverable | Gate |
| - | ----------- | ---- |
| M0 | tokenizer, prompt render, `response_template` parser, `.items()` fix | golden strings incl. full tool round trip; no GPU |
| M1 | text bf16, flat KV, streaming chat | logit parity vs HF |
| M2 | image path end to end | image parity; both wrapper tokens present; N+2 accounting |
| M3 | paged adapter | byte-equal vs flat, **then** A/B before defaulting on |
| M4 | cold tier: sidecar + both allowlists + drift guard | restore parity, process-restart test |
| M5 | `mlx convert` recipe | quantized vs bf16 quality check |

M0 first is deliberate: it needs no GPU, it unblocks every later test's prompt path, and
it is where the one hard error lives.

### Notes on M3 and M5

**M3 is not free.** `32Q/2KV/head_dim 128` matches **no** specialized paged kernel:
`grouped_qwen35` needs head_size 256 with `(24,4)|(16,2)`, `grouped_d512` needs 512, and
gemma4's crossover allowlist is `(16,1)|(16,2)|(32,4)`. It runs the generic V1/V2 kernel
with zero measured data, so default-on is an A/B decision, not a default. Set
`paged_cache_memory_mb` explicitly — pool bytes sit outside the MLX cache budget and
oversizing tanks long-context decode ~10x via residency thrash. Never auto-size to 131072.
gemma4's paged constructor hard-errors if spec grouping yields more than one
full-attention KV group; Muse-Glimmer's 13 full layers are uniform (2 kv heads, 128, bf16)
so they group to one.

**M5 needs a new recipe.** None of the existing ones fit: `nvidia` is family-gated,
fixed `unsloth` is shape-gated, `qwen3_5` leaves both 2.69 GB vocab tensors bf16, and
`mixed_*` cannot distinguish `self_attn.gate_proj [4096,6656]` from
`mlp.gate_proj [19968,6656]`. Byte budget:

| bucket | GB bf16 | share |
| ------ | ------: | ----: |
| text_body | 50.33 | 84.5% |
| vision_tower | 3.71 | 6.2% |
| embed_tokens | 2.69 | 4.5% |
| lm_head (untied) | 2.69 | 4.5% |
| projector | 0.14 | 0.2% |
| text norms (209 tensors) | 0.00 | 0.0% |

Quantize `text_body`; keep norms, both vocab tensors, the vision tower and the projector
high precision. Also check `normalize_override_key` (`utils/mod.rs:27-35`): it strips
`model.` then prefixes `language_model.model.`, so raw-HF text keys double-prefix and
per-tensor overrides silently miss, falling back to the top-level bit width.

## 9. Capacity

59.55 GB weights on 128 GB unified leaves roughly 22 GB of allocator freelist by the
`cache_limit.rs:79-96` formula. KV is cheap thanks to 2 KV heads: 13 full layers at 131072
tokens = 1.745 GB, 39 sliding layers capped at 2048 = 81.8 MB. The pressure is elsewhere:
202048-wide prefill logits, and `lm_head` being a 2.69 GB read **per decode step** — decode
may be lm_head-bandwidth-bound before anything else. Vision prefill at the 4096-token cap
means 16,384 patches, and the 13 full vision layers run 16k x 16k non-causal attention at
1536 dim; expose `max_image_tokens` with a lower default.

Two testing corollaries: the window path is **dead** for any image <= 448 px (single
window), so a small-image suite exercises none of it; and a per-layer `cu_seqlens.tolist()`
would be 50 device-to-host syncs per image — precompute once.

## 10. Deferred: video

Recorded so a later pass is cheap. `video_token_id 200091`; `grid_t = frames // 2` with
last-frame repeat padding; 2 real frames per patch, temporal-major; per-group
`Time: {ts:.1f}s` prose markers between `<|vid_start|>` and `<|vid_end|>` with
`<|vid_frame_separator|>` between groups; fps 2.0, <= 96 frames floored to even, linspace
sampling, 144-token per-frame cap. Needs a **third** media bit threaded through
`MediaCapabilities` (`engine/plan.rs:19-22` has only images/audio) -> `MediaPlan` ->
`MediaInputs` -> `TurnRequest` -> `TurnPlan` -> both `chat_napi` guards -> `ChatMessage` ->
TS `SendOptions`.

## 11. Process constraints

- Build only in this worktree; never in the shared main checkout.
- `md5` is not a valid metallib guard — good metal builds are byte-nondeterministic (this
  worktree's build differs from main's at identical size, 168,350,856 B). Use the 4
  behavioural canaries; baseline here is 4/4 passing.
- CI runs `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
- Never `vp fmt` repo-wide; it guts the vendored ggml anchor.
- `yarn typecheck` before pushing; TS changes need `tsc`, not just vitest.
