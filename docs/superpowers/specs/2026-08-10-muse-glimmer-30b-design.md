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
- Because the global layers are NoPE, their KV is **position independent** *as arithmetic*.
  This is **not actionable in the cache**: block hashes are chained over the prefix, so a
  cached block stays bound to the offset it was computed at regardless of layer kind. See the
  NoPE row of the hybrid-KV rules in §8.
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
| `config.rs` | `MuseGlimmerConfig`, text/vision sub-configs, layer-kind + NoPE tables, fail-closed asserts (both configs, including the vision tower's divisors) |
| `kv_cache.rs` | the batching seam: per-layer specs, the two-group contract, sliding pool reservations, and the **window-carrying contract** every paged dispatch must clear (§8) |
| `model.rs` | load, forward (flat + paged), `logits_tail()` |
| `attention.rs` | qk-norm x 3.87, optional RoPE, sigmoid output gate |
| `decoder_layer.rs` | 4-norm block, dual epsilon |
| `mlp.rs` | SwiGLU |
| `layer_cache.rs` | per-layer kind -> `RotatingKVCache` / paged / flat |
| `vision.rs`, `vision_embedder.rs`, `vision_window.rs` | tower, patch embed + zero-pad resample, window index |
| `image_processor.rs` | token-capped smart_resize, LANCZOS, `(t,c,ph,pw)` flatten |
| `output_parser.rs` | `response_template`-driven ATEM parser; `terminators` = the text fields' closes UNIONED with `tokenizer_config.json`'s `eos_token` |
| `stream_guard.rs` | per-turn streaming safety: channel routing, byte-exact header grammar, provenance-gated scan (`authority_spans`), `TurnEnd` |
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

13. **miniJinja cannot parse a ternary as a call keyword argument** — so the template does
    not compile at all, and every render fails, including one that never reaches the tool
    path. `parse_expr_noif` (`= parse_or`, `minijinja-2.23.0/src/compiler/parser.rs:671`)
    cannot consume a trailing `if`, and the argument loop then meets the identifier `if`
    where it demands `,` or `)`. Minimal repro:
    `{% set n = namespace(name=a if a else '') %}` → `syntax error: unexpected identifier,
    expected ','`. The checkpoint's tool-name fallback is exactly that shape, at byte 5737.
    Python Jinja2 3.1.6 accepts both spellings, so the checkpoint is well-formed and
    miniJinja is stricter. There is no syntax option, 2.23.0 is the newest release, the
    dependency is plain crates.io with no `[patch]`. The restriction is general to all call
    kwargs but narrow to the **ternary** — filters in kwargs parse fine, which is why
    gemma4's `namespace(name=… | default(…))` works today. Fix: a region-aware source
    transform that parenthesises the value, modelled on `neutralize_generation_tags`
    (`tokenizer.rs:381`) and applied beside it at `:1401`; gated on byte-identity across
    every other installed template. A `str::replace` would corrupt a ternary inside a
    string literal, `{% raw %}`, or a comment.
14. **`.get(key)` returns `Undefined` where Python returns `None`** (`tokenizer.rs:1304`),
    so `{% if x is none %}` never fires on a missing key. This template gates its
    `end_turn` default on exactly that, so **every plain assistant turn terminates
    `<|eom|>` (0x6D) instead of `<|eot|>` (0x74)** — off-distribution history on every
    multi-turn chat. `Value::from(())` matches Python; `Value::UNDEFINED` matches nothing.
    Measured blast radius: gemma4 and Muse-Glimmer only. Qwen3/3.5/3.6/ASR, Ornith,
    AgentWorld, agents-a1 and LFM2.5-1.2B have no `.get(` at all; LFM2.5-2.6B/8B use it
    only for truthiness, `==` and kind-tests; qianfan / PaddleOCR-VL / Harrier ship no
    cached template. No template anywhere applies `is defined` / `is undefined` to a
    `.get()` result. gemma4 argues *for* the fix: our serializer never emits a top-level
    `name`, so its `follow.get('name')` misses on every tool round trip and survival rides
    on the `tc.id == tcid` conjunct — post-fix miniJinja is 1:1 with HF on all five
    id-presence cases, converting quiet prompt corruption into a loud failure. Ship the
    upstream `name` normalisation with it. `map_get_bridge_mirrors_python_dict_get`
    (`:3195`) asserts a value that contradicts Python and must be corrected, not preserved.

Traps 13 and 14 were found by writing the M0 golden gate and then confirmed independently
with minimal reproductions. Nothing in this repo differentially byte-diffs miniJinja against
Python Jinja2 over the installed templates; that harness would have caught both, and is
worth building outside this project.

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

**The whole TypeScript surface lands in M1, in one commit, and cannot land earlier.**
`ModelFamilyDescriptor.nativeModelClass` is a required field (`model-loader.ts:79`) and
`LoadableModel = InstanceType<RegisteredModelFamily['nativeModelClass']>` (`:196`), so
whatever fills it enters the exported public type union. Before the native class exists, a
registry row costs either an uninstantiable placeholder in that union — with
`discoverModels` then advertising a family whose `load` throws — or a weakening of the
registry's types. `ModelType` derives from the registry (`:183`), so the
`Record<ModelType, …>` key, `LAUNCH_PRESETS`, `FAMILY_TRAITS` and `supportsImages` all
depend on the row and cannot precede it. Until then the behaviour is already correct and
loud: `Unsupported model_type "muse_glimmer"` (`:312`). M1 adds all five sites together.

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
- **the window-carrying contract** (`muse_glimmer/kv_cache.rs`): a sliding group on a
  window-blind route is an `Err`; no admitted sliding dispatch can present a window of `0`
  through either accessor, enumerated over all three `WindowCarrier`s rather than spot-checked;
  a full group asks for no mask so the causal fast path survives; the window tracks the config
  and is not a constant; a dispatch plan that forgets or permutes a group is refused. Plus the
  source tripwire that fires the moment paged wiring appears in this family without naming the
  seam. Contract and reasons in §8

Already landed, outside this family, and the reason it belongs on this list: the
"sliding adapter x attention numerics" cell that no test occupied is now occupied —
`sliding_window_masks_retired_positions_on_every_prefill_route`
(`transformer/paged_kv_cache_adapter.rs`, committed `179e826d`), pure Rust, no weights, not
`#[ignore]`d, skips cleanly without Metal. The **fix** it guards landed too — `eb5713e3`,
`7a898db4`, `f28e99f4` (gemma4) and `c0faba4a` (the model-independent adapter refusal); full
status table in §8. Gemma4 is now the reference implementation to copy, not a broken template. It supersedes the "cheapest confirming test" this
spec used to propose, whose prescribed `block_size 4` was **impossible**: the Metal kernel
instantiates only block_size 8 / 16 / 32, and the C++ validator checks only `block_size > 0`, so
a block_size-4 dispatch fails at kernel LOOKUP and an implementer would read the null return as
their own bug. The landed test uses a 2x-scaled geometry (block 8, window 16, 48 tokens, prefix
32) preserving every ratio.

Tier 2 — real weights, local `#[ignore]`:

- logit parity vs an HF reference dump (fixture generated once, committed if small)
- image parity on a fixed image
- `muse_glimmer_paged_vs_flat_parity` — byte-equal greedy, fresh and delta. Run it as a **short
  sweep past the onset, not one long prompt**: onset+ε, onset+2 chunks, onset+4 chunks (this
  family's sliding over-attention onset is ~2050 tokens, §8). A short-prompt-only gate is green
  for the wrong reason — gemma4's equivalent matched flat at 1307 tokens and diverged at 1374 —
  but so is a single long prompt, and that is the stronger argument for the sweep: divergence is
  **non-monotonic in prompt length**. In gemma4's route-forced runs every route matched flat at
  2771 tokens while the defect was live, between mismatches at 1374-2066 and again at 4008. One
  lucky length reads as proof
- the real-weights **golden gate for the prompt surface** stays a local `#[ignore]` and that is
  not a preference: the checkpoint is 59.5 GB and will not fit a macos-26 runner
  (`ci.yml:248-253`, the documented qwen3_5-MoE position). Nothing in CI can replace it, which
  is why the Tier 1 list above has to carry the arithmetic
- cold-tier restore parity, single-gate binaries only
- `__test__/models/muse-glimmer-concurrency-e2e.test.ts` — the four-assertion continuous-batching
  gate, of which only `fusedGreedyEpilogueSteps > 0` is non-vacuous. Contract in §8

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
it is where the one hard error lives. It found two more (traps 13 and 14).

### M1 requirements discovered while building M0

These are not optional polish. Each came out of an adversarial round on M0 code and cannot
be settled inside M0.

- **Token provenance must reach the output parser — LANDED IN M0, both halves. Nothing left
  for M1.** Kept as the rationale, not as a task: `StreamGuard::new` resolves `CONTROL_MARKERS`
  to ids through the checkpoint tokenizer and accumulates `control_spans`, and
  `GeneratedTurn::from_guard` is the only `pub` constructor — `from_token_spans` is
  module-private, so a sibling (`stream_guard`, or M1's `model.rs`) cannot name it at all. That
  privacy IS the boundary, and it is stronger than the `pub(super)` version an earlier round
  shipped. Do not re-implement span building in M1. The finding that produced it: `<|eot|>` rendered from
  real token 200008 and `<|eot|>` written as seven literal characters are *the same bytes*
  once a `&str` reaches the parser, so a message terminator quoted inside an answer let the
  anchored tool call after it execute — `rm {path:"/"}` out of explanatory prose. Every
  signal the parser could add is itself made of those bytes, so the fix is a signature, not
  a rule: the parser takes the byte ranges where a genuine special-token rendering
  occurred, and treats a terminator or an anchor as a boundary only inside one of them.
  Whatever consumes the token stream — the streaming guard is the natural place — has to
  build those spans, because it is the last layer that still has token ids. A parser that
  can emit an executable call from an un-provenanced string is the wrong shape regardless
  of how careful its rules are.
- **The guard's SCAN is provenance-gated too, not just what it reports — LANDED IN M0.**
  This was recorded as "pinned, not fixed" for one round and is now FIXED, because two
  independent reviews flagged the same class of problem and the second one found a case the
  first did not. The defect: `StreamGuard::scan` matched decoded TEXT while
  `output_parser` decided on spans, so on byte-identical input the guard cut the stream
  where the parser reported prose — and, worse, opened a channel where the parser opened
  none. Measured on the REAL vocabulary with ordinary sampleable ids (`<`=40, `|`=104,
  `|>`=159276), a typed `<|eom|><|start|>assistant to=user<|message|>` inside a `to=self`
  message made the guard STREAM the chain of thought as the answer, byte for byte the text
  the parser puts in `reasoning` with `content = None`. Second shape, not in the first
  review: a real `<|eom|>` + a TYPED anchor + ONE real `<|message|>` forged a content
  header, because `classify_header` is handed only the region text and never provenance.
  Neither is privilege escalation — `tool_calls` stayed empty in every probe, since the
  parser's action gates are all intact — but chain-of-thought reaching the user-visible
  channel is exactly what `Channel::Reasoning` exists to prevent. The rule now:
  **a control literal exerts structural authority only where a control token put those
  bytes there, or where ONE id's own text contained them; the ATEM literals stay decided
  by the recipient.** `authority_spans` is a guard-private SUPERSET of the parser's
  `control_spans` and nothing is ever added to the latter — the second clause is what keeps
  a single token spelling `<|eot|>` plus trailing text stopping the turn, without resting on
  the checkpoint's vocabulary property (now pinned in the gated tier: 202,048 keys, zero
  carriers). Each of the four gates mirrors a rule the parser already had —
  `earliest_terminator`, `next_anchor`, `tool_header_at` — and text headers stay ungated on
  BOTH sides, which is the standing ruling that actions are gated end to end and text is
  not. **M1 must not re-decide this**: it is the streaming safety boundary, and its tests
  are written against the resolved ruling.
- **A call's completeness is `</atem:invoke>`, not its message's terminator — LANDED IN M0
  as documentation, tests and a signal; the parser's logic is UNCHANGED and deliberately
  so.** Two review rounds in a row asked `collect_tool_calls` to refuse calls from a
  message whose terminator never arrived, both misreading `terminators`' doc ("will act
  AFTER") as "will act ON". The behaviour they describe is real — a `max_tokens` seal right
  after `</atem:invoke>` does yield a call from an unterminated message — but the fix is a
  correctness REGRESSION: measured, it discards 2 complete calls after a cap seal and 1
  complete call from a tool message ended by `<|end_of_text|>`, a turn that is not
  truncated at all. The per-CALL rule already there is sufficient rather than merely
  cautious, and the proof is `StreamGuard.raw` being append-only (`stop`/`seal`/`flush`
  never shorten it, cap-refused ids are refused before decoding, mid-scalar bytes stay in
  `decode_ids`): `raw` is a strict PREFIX of the model's output, so a present
  `</atem:invoke>` proves the whole invoke arrived and truncation cannot produce a
  closed-but-partial call. What was actually missing was a SIGNAL — `GuardOutcome::EndTurn`
  was identical for a natural `<|eot|>` and a cap seal — so `StreamGuard::turn_end`
  -> `TurnEnd` now says WHY a turn ended. **If M1 wants to be conservative about
  truncation, it holds back the LAST call on `TokenCap`; it never drops the list, and never
  in the parser, which cannot see finish_reason and for which the terminator set is not a
  sound proxy.**
- **`terminators` is derived from TWO shipped sources, and must stay a union — LANDED IN
  M0.** `response_template`'s `close` lists describe what `chat_template.jinja` RENDERS
  (`<|eot|>` 6x, `<|eom|>` 4x, `<|end_of_text|>` never), so they cannot name a
  sampling-time stop. Before the fix a `to=user` answer ended on 200001 parsed to
  `content = Some("The answer is 42.<|end_of_text|>")` while the guard streamed
  `"The answer is 42."`, and `reasoning` had the same leak with no clean copy anywhere
  since the guard never streams that channel. The extra terminator comes from
  `tokenizer_config.json`'s top-level `eos_token` — already a marker STRING in the body the
  parser parses, so no id->text resolution and no tokenizer. It must stay a UNION: that key
  names 200001 and OMITS 200008, which `engine/persistence.rs` already documents as a trap
  that makes a loader never stop. Absent stays legal; empty is refused.
- **A tool call is recognised only where the recipient is a tool channel, and every accepted
  invoke name must equal that recipient — LANDED IN M0.** Enforced in `output_parser` (the
  invoke name is compared to `recipient` and the scan stops on a mismatch) and pinned by
  `an_invoke_name_must_equal_its_recipient` plus the quoted-header case. Rationale retained
  because it supersedes an earlier reading that
  put the check at dispatch. Dispatcher validation cannot recover a discarded recipient: it
  sees `rm`, which may be legitimately declared, and cannot know the message was addressed
  to `userX`. Equality is protocol fidelity rather than a restriction — `chat_template.jinja`
  builds `'<|start|>assistant to=' + tc.function.name` and `render_atem`'s invoke tag from
  the *same string*, one call per message. The spec's `repeats: true` permits repeated
  matches and repeated messages; it does not license a different invoke name under one
  recipient. Keep registry validation at dispatch as defence in depth, not as the control.
- **The guard takes decoded tokens, not a count and not a chunk.** Inferring the count from
  decoded character length was wrong, not merely imprecise: **70 decoded tokens exceed the
  old 64-character assumption** (id 169871 at 113 characters, id 162250 at 112), and the
  guard truncated a legitimate 560-character answer to 448. Note the measurement trap —
  74 *raw BPE lexemes* exceed 64 characters but only 70 *decoded* tokens do, and counting
  vocabulary keys instead of decoded output is what produced the wrong figure first time.
  Trusting a caller-supplied number is not enforcement either: a slice with one entry per
  token is, which is why `push(&[&str])` and `push_token(&str)` replaced it.
- **Size the header allowance from the longest configured recipient — LANDED IN M0.**
  `StreamGuard::new(.., max_recipient_chars)` derives
  `header_allowance = len(ANCHORED_HEADER_PREFIX) + max(max_recipient_chars, BUILTIN_RECIPIENT_CHARS)`,
  clamped by `ABSOLUTE_MAX_HEADER_CHARS` — the clamp **is** the "separate absolute memory
  bound" below. Rationale retained: a fixed 128 was the
  same mistake as the fixed 64: a 107-character advertised tool name produces a
  129-character anchored header, and the same legal name was accepted or refused depending
  on which header carried it. `FunctionDefinition` accepts unrestricted strings, so either
  derive the allowance per turn or document and enforce a maximum before rendering. Keep a
  separate absolute memory bound.
- **Wire the session-pinned render options into the production chat path.** STILL OPEN,
  re-verified: `apply_chat_template_sync` **and**
  `apply_chat_template_sync_with_content_order` both hard-code
  `RenderContextOptions::default()`, so no family's production call site can supply
  `current_date`. M0 pins it only through the render entry point the golden tests use.
  Mechanically this is one options-carrying overload. **The blocker is the consumer, not the
  plumbing:** there is no muse_glimmer production render site to pin yet, and adding a 7th
  parameter to a function nine families call is a shared-file change. Land it in the same M1
  commit as this family's first real render, so the new parameter has a caller that proves it.
- **`ChatMessage` cannot carry `recipient` or `end_turn`.** STILL OPEN, re-verified: neither
  field exists in `engine/backend.rs`. The template reads both via `.get`, so multi-part
  assistant turns and non-`user` recipients cannot be replayed. **Blocker:** both fields change
  the `#[napi(object)]` surface, which enters the exported TS type union — and per §6 the whole
  TS registry surface cannot land before `nativeModelClass` exists. So this is coupled to M1's
  one-commit TS landing, not independently shippable.
- **The whole TypeScript registry surface lands here, in one commit** — see §6.

### The continuous-batching contract — M1 builds into it, M3 completes it

`37f1d68e` ("Complete continuous batching through Stage 2", #116, 172 files) landed a real
scheduler trait. Every one of the five existing families was **retrofitted** onto it after
its forward pass already existed; Muse-Glimmer is the first that can be written against it.
Joining late costs a second pass over `model.rs`, `attention.rs` and the whole prefill loop,
which is why the contract is recorded here rather than discovered at M3.

It is a trait, not a `match` on a family enum:

```
HybridSchedulerCommand  (hybrid_scheduler.rs:70)   on MuseGlimmerCmd
HybridSchedulerBackend  (hybrid_scheduler.rs:81)   on MuseGlimmerInner
  : PagedBackend        (engine/backend.rs:1151)
  : ChatBackend         (engine/backend.rs:605)
```

All three are `pub(crate)` — in-crate only, no NAPI surface of their own.

#### Mandatory members: omit one and the crate does not compile

**Eleven** methods, one const and four associated types have no default body. (A count of
"12 methods" folds in the const; there are 11 `fn`s.)

| Required member | line | Why it constrains the M1 forward pass |
| --------------- | ---: | ------------------------------------- |
| `type Command` / `RestoreTicket` / `OwnerState` / `StepExecutor<'a>` | 82-88 | `OwnerState: Default` means per-owner state must be constructible empty — no "must have run a turn first" invariant |
| `const SCHEDULER_NAME` | 89 | Interpolated into every scheduler error message; make it `"Muse-Glimmer"` so the `supports_delta` refusal below is attributable |
| `paged_adapter` / `paged_adapter_mut` | 94/95 | Return `Option<&PagedKVCacheAdapter>` — **one** adapter. A hybrid returns its *full* group's adapter here and coordinates the rest itself, exactly as gemma4 does |
| `max_position_embeddings` | 178 | Scheduler clamps every turn's output budget against it (`hybrid_scheduler.rs:1437-1440`, `min`'d with `scheduler_per_seq_context()`). Return `text_config.max_position_embeddings` — the field is parsed and validated as of `c8287d3b` (`MuseGlimmerTextConfig`, `config.rs:142`); note the trait returns `i32` while the config holds `usize`, and the scheduler's `unwrap_or(1).max(1)` means a bad conversion degrades to a 1-token context rather than erroring |
| `activate_paged_seq` | 194 | Rows are per-`SeqId`; the forward pass may not assume one live request |
| `run_paged_decode_step_batched` | 199 | The single hardest requirement — see the shape rule below |
| `replace_cached_token_history` | 200 | The scheduler swaps whole token histories between rows; a forward pass that caches a `Vec<u32>` privately must expose the swap |
| `owner_tokens` | 201 | Static fn on `OwnerState`, no `&self` — owner history may not live inside the model |
| `capture_owner_state` | 206 | **Not** preemption. Its only call site in the crate is turn completion with the owner keeping its cache: `finish_completed` (`hybrid_scheduler.rs:2124`) calls it at `:2187-2191`, guarded by `outcome.is_ok() && turn.payload.reuse_cache`. `grep` returns exactly two hits, that call and the trait decl. Preemption (`:1960-1975`) never touches it — it calls `preempt_scheduled_cache`, and the state comes BACK via `install_owner_state` (`:1404/:1641/:2040/:2142`). So the row is **not** being evicted: capture what the next turn on this owner must see, and do not skip state the live turn still needs. It runs on the "mlx-model" thread right after a completed turn, so keep it cheap — but "no GPU" is a latency preference at that site, not a hard constraint the way it would be during eviction |
| `build_scheduled_prefix` | 218 | Constructs a `PrefixState` for an arbitrary `(cached_prefix_len, suffix_len, first_chunk)` triple. This is where a hybrid's *sliding* re-prefill boundary has to be expressible, not just the full group's |
| `step_executor` | 281 | |
| `execute_barrier` | 282 | Family-specific commands (media, convert, save) stay ordered barriers |

`run_paged_decode_step_batched` **must return `[N, 1, vocab]` with `N == rows.len()` and row
order preserved.** The shape is not in the signature, and **nothing downstream enforces it.**
There is exactly ONE enforcement point and you have to write it: assert the shape inside
`run_paged_decode_step_batched`. It **cannot** be written before M1 — the blocker is not
convention, it is that `MuseGlimmerInner` and its `HybridSchedulerBackend` impl do not exist, so
there is no function body to put the assert in and nothing to call it from. It is therefore an
M1 line item rather than a deferred nicety, and the two facts below are why nothing else will
catch it for you.

- `engine/batch_sampling.rs:46-53` — `batch_greedy_tokens` errors on `ndim != 3 || shape[1] != 1`,
  but that `Err` is a **soft degrade, not enforcement**. Its only production caller is
  `batch_greedy_tokens_or_fallback` (`batch_sampling.rs:72-84`, called at
  `hybrid_scheduler.rs:901`), which downgrades the error to a `tracing::warn!` and returns
  `None` — by design, so a shared optimization cannot fail a wave whose forward pass
  succeeded. Return `[N, vocab]` or `[1, N, vocab]` and the wave completes with
  correct-but-scalar sampling, `fusedGreedyEpilogueSteps` stays 0, and nothing fails. That is
  precisely why gate (d) — asserting the fused epilogue actually engaged — is the only
  non-vacuous assertion available here.
- `hybrid_scheduler.rs:958` — `logits.slice_axis(0, index, index+1)` indexes by **row
  position**. A permuted or short batch dimension does not error; it silently hands row *i*'s
  logits to row *j*.

Seven recurrent-state hooks (`hybrid_scheduler.rs:179-198`) are defaulted no-ops. Muse-Glimmer
is pure-KV (no GDN, no conv state), so it omits all seven; `recurrent_state_bytes() == 0`
means the scheduler charges it zero recurrent memory. That is correct here and must not be
copied by anything with a recurrent lane.

#### The one hard error: `supports_delta` must be `true`

Everything else in this contract fails *closed by running serially*. This one hard-errors.

```
execution_plan() -> PagedAttentionPlan { supports_delta: false }
  -> TurnPlan::resolve: use_paged_attention = !is_delta || supports_delta   (plan.rs:236)
  -> a DELTA turn resolves to TurnPath::Flat
  -> hybrid_scheduler.rs:1417-1433 refuses: "<NAME> scheduler only admits plain
     text paged autoregressive turns"
```

Turn 1 succeeds, turn 2 of every conversation fails. All five current families set
`supports_delta: true`; the only `false` in the tree is a unit test (`plan.rs:350`). This is
a **requirement**, not a note: M1 must set it true and M3 must keep it true.

#### One adapter per group — not one adapter with per-kind tables

```
MuseGlimmerKVCacheCoordinator
├─ group 0  AttentionKind::Full              13 layers  {3,7,…,51}
│    PagedKVCacheAdapter::new(alloc0, pool0, block_size)
│    own BlockAllocator + own LayerKVPool
└─ group 1  AttentionKind::SlidingWindow{2048}  39 layers
     PagedKVCacheAdapter::new_sliding(alloc1, pool1, block_size, 2048, max_seq_len)
     own BlockAllocator + own LayerKVPool
```

Verified against gemma4's constructor (`gemma4/model.rs:2262-2292`): a fresh
`BlockAllocator` and `LayerKVPool` per group, then `new` vs `new_sliding`. **The two spans use
the same Metal kernel**; the entire difference at dispatch is one integer, the kernel's
`sliding_window: i32`. That is why the trap in the next-but-one subsection is a dropped argument
rather than a missing code path — and why the *sign* of that integer is load-bearing too: the
window is range-checked into an `i32` at admission rather than cast there, because a `u32 as i32`
in that slot is how a 3-billion-token config window became `-1294967296`. See the
`sliding_window` validation subsection.

Grouping is structural, not a heuristic. `group_layer_kv_cache_specs`
(`transformer/kv_cache_spec.rs:504`) keys a `BTreeMap` on **`(attention_kind, physical_layout)`**
and `group_id` is the sorted enumeration index. `AttentionKind` declares `Full` before
`SlidingWindow` and derives `Ord` (`:41-46`). Muse-Glimmer's head geometry is **uniform** —
one `head_dim 128`, one `num_key_value_heads 2`, no global overrides, no `k_eq_v`, so one
`KVCachePhysicalLayout` for all 52 layers. Therefore:

- **exactly 2 groups**, and `group_id 0` is **Full**. Do not write code that discovers this
  at runtime and do not assume the reverse ordering. This is safe to hard-code **because
  `compute_layer_kv_cache_groups` fails closed when it cannot hold**: a single-kind
  `layer_types` table parses cleanly (the config's NoPE↔Full biconditional only fires when the
  two tables *disagree*, and a uniform table agrees with itself) and would collapse grouping to
  ONE group. All-sliding is the silent direction — `groups[0]`, whose adapter is returned from
  `paged_adapter()` and publishes into the content-addressed prefix cache, would be the
  *sliding* group, and a sliding block's contents depend on where the window was when it was
  written. `muse_glimmer/kv_cache.rs` now refuses any grouping without at least one group of
  each kind, naming the observed counts. It deliberately does **not** enforce the
  `[S,S,S,F] × 13` pattern: a future hybrid ratio is still a hybrid.
- gemma4's ">1 full-attention group" refusal (`gemma4/model.rs:2100-2107`) is **structurally
  satisfied**, not merely untriggered. No special handling needed, and none of the
  `effective_head_dim(is_global)` / `effective_kv_heads(is_global)` indirection gemma4 carries
  (`gemma4/model.rs:8106-8118`) needs an equivalent — the layout is loop-invariant.
- there are **no KV-shared/aliased layers**. Every `LayerKVCacheSpec` keeps
  `shared_kv_anchor: None`, so in both groups `physical_layer_indices == layer_indices`, and
  the seam's `MissingSharedKVAnchor` / `SharedKVAnchorIsAlias` / `SharedKVIncompatible` errors
  are unreachable. Do not port gemma4's alias branch (`gemma4/model.rs:8135-8143`).

**Deliberate divergence from vLLM, recorded so nobody "fixes" it.** vLLM's grouper
(`vllm/v1/core/kv_cache_utils.py:1106-1211`) buckets on the *entire* frozen spec dataclass,
then merges, then **splits every bucket to an equal layer count**: for 39+13 it emits **4
groups of 13** (3 sliding + 1 full, zero padding since 39 % 13 == 0). It needs that because
all its groups draw block IDs from **one shared free list** with a single uniform page size
(`get_uniform_page_size` is a bare `assert len(page_sizes) == 1`). Our seam gives each group
its own `BlockAllocator` and `LayerKVPool`, so neither the equal-layer split nor page-size
unification applies. **2 groups is correct for us.** The cost we avoid is also worth knowing:
vLLM's `resolve_kv_cache_block_sizes` sets `scheduler_block_size = lcm(group block sizes)`, so
per-kind geometry there quantizes prefix-cache hits to LCM boundaries. We pay none of that
while geometry stays uniform.

#### The three inputs the seam needs

| Input | gemma4's source | Muse-Glimmer |
| ----- | --------------- | ------------ |
| `block_size` | `config.paged_block_size.unwrap_or(16)` (`gemma4/model.rs:2076`), a config knob | **caller argument.** This family's config is Rust-internal with no NAPI surface by design (`config.rs`), so the knob has to come from the paged-adapter construction site, not the checkpoint |
| `cache_dtype` | `KVCacheDType::BFloat16` (`gemma4/model.rs:2085-2091`) | same — checkpoint dtype is `bfloat16`; keep it a caller argument |
| `max_model_len` | `u32::try_from(config.max_position_embeddings)` (`gemma4/model.rs:8158`), a **required** config field | `text_config.max_position_embeddings` (`MuseGlimmerTextConfig`, `config.rs:142`), **landed in `c8287d3b`** — required, no `#[serde(default)]`, validated non-zero and u32-fitting in `from_json_str`'s `max_position_embeddings` arm (`config.rs:451-464`). Consumed by `compute_layer_kv_cache_groups`, which re-checks both halves (`kv_cache.rs:199-212`). **Cite by symbol, not line:** this family's `config.rs` and `kv_cache.rs` are still churning, and every line pair above was already stale once — `config.rs:346-358` now lands on the `num_hidden_layers == 0` guard, which looks like a plausible validation and has no u32 clause |

**Landed in `c8287d3b`; the reasoning below is kept as the rationale for why it must not be
defaulted.** `RawTextConfig` (`config.rs:65`) and `MuseGlimmerTextConfig` (`config.rs:142`) both
carry `max_position_embeddings` as a **required** field with **no `#[serde(default)]`**, copied
through in the validated `Ok(Self { .. })` block. Do not re-add it, and do not add a second
spelling. A default would be exactly the silent trap the module's own
`defaulted_fields_are_read_from_the_file_when_present` test exists to catch: an absent key and
a 131072 key would then be indistinguishable, and the sliding pool would be sized off a number
nobody wrote. Validation is in `from_json_str`'s `max_position_embeddings` arm
(`config.rs:451-464`) — non-zero (a 0 makes the full-attention
bound `div_ceil(0, block_size) == 0`, a group that admits nothing) and u32-fitting — and
`kv_cache.rs` re-checks **both** halves, because it is `pub` and must not trust a config
assembled elsewhere. It is **not** on `MuseGlimmerVisionConfig` — the vision tower's own
`layer_types` and `max_position_embeddings` are deliberately unparsed, and the KV-spec seam is
**text-decoder only**.

That "no cross-field resolution" note on `MuseGlimmerVisionConfig` had become a trap of its own
and is now closed. It was literally true and read as "nothing here needs checking", so
`patch_size: 0`, `merge_size: 0`, `patch_temporal: 0`, `pos_emb_height/width: 0` and
`num_attention_heads: 0` all parsed and reached the runtime while the text config next door
refused seven separate traps. `MuseGlimmerVisionConfig::validate` now refuses every zero and
enforces `hidden_size % num_attention_heads == 0` (1536 / 16 = 96), and `from_json_str` calls
it. The reason each field matters is on the method, not here, but note the head rule is
deliberately the **opposite** of the text decoder's: the tower has no `head_dim` key, so its head
dim IS that quotient, whereas the decoder's `head_dim` is independent and `32 x 128 != 6656` on
purpose. Copying either rule to the other side is a bug.

`sliding_window` validation is landed (`3c0c6859`, upper bound corrected in this round): the
config refuses `0` and refuses a value **past `i32::MAX`** at load. Keep that placement.

**The bound is `i32::MAX`, not `u32::MAX`, and the reason is downstream of every type on the
path.** `AttentionKind::SlidingWindow` stores the window as a `u32`, so `u32::MAX` reads as the
natural bound and was the one originally shipped. But both things that consume the window take an
`i32` — the Metal kernel's `sliding_window` FFI argument, and `create_causal_mask`'s `window_size`
(`array/mask.rs:35`). So the band `[2^31, 2^32-1]` fits serde's `usize`, fits the config guard,
fits the `u32` spec payload, used to fit `PagedWindowSlot`, and still arrived at a dispatch **negative**.
Measured end to end on this seam before the fix, with `"sliding_window": 3000000000`:

| stage | before | after |
| ----- | ------ | ----- |
| serde -> `usize` | `3000000000` | same |
| `from_json_str` guard | `Ok` (`u32::try_from`) | **`Err`**, naming the field and `i32::MAX` |
| `compute_layer_kv_cache_specs` guard | `Ok`, 39 sliding specs | **`Err`** |
| `sliding_window_max_admission_blocks` | `Ok(8193)` — caps, see below | unreachable |
| `PagedWindowSlot::kernel_slot()` | `Some(-1294967296)` | **`Err`** at admission |
| `PagedWindowSlot::mask_window()` | `Some(-1294967296)` | **`Err`** at admission |

The two routes then diverged, and only one of them was loud — which is why the fix is described
honestly as two different things:

- **`KernelArgument` already failed CLOSED.** The C++ validator rejects a negative window
  (`mlx_paged_ops.cpp:483`, *"must be >= 0 (use 0 to disable the sliding mask)"*), the extern-C
  entry point catches it and returns `nullptr`, and `check_handle` turns that into an `Err`. For
  this route the new refusal is a **diagnostics** improvement only: same outcome, named at the
  config field instead of six frames into the FFI. Do not oversell it as a security fix.
- **`ExplicitMask` failed OPEN and SILENT, and that is the half worth closing.**
  `create_causal_mask` keeps a cell only when `linds >= rinds && linds < rinds + window_size`, so
  a negative width keeps **nothing**. Measured on real MLX: 0 of 96 cells kept, versus 68 for the
  causal baseline. An all-`false` mask fed to the *fused* `scaled_dot_product_attention` does
  **not** produce NaN and does not error — every query row comes back byte-identical to
  `mean(v, axis=keys)`, i.e. the uniform mean of every key including future positions and
  never-written null-block pages, `max|delta| 1.2485` against the windowed reference. (The naive
  unfused `where(mask, s, -inf)` + softmax path *does* give all-NaN; only the fused kernel is
  silent, and the fused kernel is the one in use.) That is the same failure shape as the gemma4
  dropped window this whole seam exists to prevent.

`PagedWindowSlot` now stores the window as the `i32` its consumers take, range-checked once in
`paged_window_for_kind`. There is no `as i32` cast left in the type, so the invariant is
structural rather than remembered — the same argument that makes the fields private.

**Do not tighten the bound further, and specifically not to `max_model_len`.** A window larger
than the context is legal and must stay expressible: gemma4 decides *"this window cannot bite,
keep the unmasked causal fast path"* by comparing the window against the context
(`cache_hit_dense_window_arg`), which requires representing it first. Note also that
`max_position_embeddings`' guard stays `u32::try_from` — that field's carrier really is a `u32`
(`max_model_len`), so narrowing it to `i32` would be a wrong bound copied from a neighbour.

**The admission cap is deliberate and must stay silent.** `sliding_window_max_admission_blocks`
does `min(sliding_window - 1 + max_chunk, max_model_len)` with saturating arithmetic, so an absurd
window returns `Ok(8193)` — the same answer as a window that exactly fills the context — with no
error. That reads like a missing guard and is not one: it is vLLM's rule line for line
(`kv_cache_interface.py:618`, `num_tokens = min(self.sliding_window - 1 +
max_in_flight_tokens, max_model_len)` then `cdiv(...) + 1`), and turning it into an `Err` would
refuse the legal "window wider than the context" configuration above. Pinned by
`kv_cache_spec::tests::a_sliding_window_wider_than_the_context_caps_silently_and_is_not_an_error`,
which also pins that this function cannot enforce the `i32` bound itself: it never sees a config
field, so its error could not name one.

Note the seam's own refusal is weaker than it looks.
`KVCacheSpecError::InvalidSlidingWindow` is raised **only** inside
`AttentionKind::sliding_window_max_admission_blocks` — i.e. in the admission-block
**arithmetic**, not in validation. `validate_layer_kv_cache_specs` checks duplicate layer
indices, layout validity and shared-anchor rules and **never inspects the window**, so the
`KVCacheCoordinator::from_groups` -> `derive_layer_kv_cache_routes_from_groups` path, which
calls only the `validate_*` functions, catches a zero window nowhere at all. Not live today
(both families guard the window themselves) but the seam is `pub`. Earlier revisions of this
line and of `config.rs`' comments said the refusal lived in validation "but late"; that
mislocates it, and a reader relying on it would drop the config-level check. gemma4 validates at
`model.rs:8096-8101` instead of in its config, which is the weaker placement.

#### Sliding pool sizing is a function of the PREFILL CHUNK

```
AttentionKind::sliding_window_max_admission_blocks           (kv_cache_spec.rs:64-79)
  = div_ceil( min(window - 1 + max_chunk, max_model_len), block_size ) + 1
```

It is **not** `window x max_num_seqs`. The `+1` is load-bearing: the live window's token range
is not block-aligned, so it straddles one extra block, and `prune_sliding_window_for` frees
only whole blocks below `cutoff / block_size`. Without the `+1` the blocks actually held
exceed the reservation, which is a mid-prefill OOM or an admission deadlock. vLLM's
`ChunkedLocalAttentionSpec` has the same formula with **no** `+1` precisely because chunk
boundaries *are* block-aligned — that contrast is the proof of what the `+1` buys.

Concrete, at `block_size 16`, `max_model_len 131072`, `window 2048`:

| group | blocks/request | tokens covered | KV bytes (bf16, 2 kv heads x 128) |
| ----- | -------------: | -------------: | --------------------------------: |
| full (13 layers) | 8192 | 131072 | 1.745 GB |
| sliding (39 layers), `max_chunk` 512 | **161** | 2576 | 102.9 MB |
| sliding, `max_chunk` 1024 | 193 | 3088 | 123.3 MB |
| sliding, `max_chunk` 2048 | 257 | 4112 | 164.2 MB |
| dense 52-layer equivalent | 8192 | 131072 | 6.979 GB |

A 50.9x block reduction on 39 of 52 layers is the whole prize, and it **scales with
`max_chunk`**: raising the prefill chunk under-provisions a pool sized for the old chunk. The
pool sizer and the runtime admission gate must therefore call **one** function — vLLM keeps
them on one source of truth for exactly this reason. The comment that names both failure modes
is `single_type_kv_cache_manager.py:178-186`, inside `get_num_blocks_to_allocate`'s admission
cap: *"Drift between the two would re-introduce the deadlock from issue #39734 or, worse,
mid-prefill OOM."* (`grep -niE "deadlock|oom"` over that file and `kv_cache_interface.py` at
`b369f10d5c` returns exactly those two lines.) The lookup that wires the single source is
`:1860-1875`, where `get_manager_for_kv_cache_spec` passes
`kv_cache_spec.max_admission_blocks_per_request(...)` — the same spec method the startup sizer
`max_memory_usage_bytes` calls — into the runtime manager. Two independently derived caps is
the bug shape.

Two more sizing facts:

- Blocks are **not fungible across groups**. A block ID in the full group costs
  13 x 16 x 1024 = 213 KB; in the sliding group it costs 39 x 16 x 1024 = 639 KB, because the
  price is per-layer-in-group. Never sum group block counts as if they were the same currency
  when setting `paged_cache_memory_mb`.
- Muse-Glimmer's ratio is **3 sliding : 1 full**, so the asymptotic hybrid/dense KV floor is
  `1/(1+3) = 25%`; at 131072 tokens with `max_chunk 512` it is **26.5%**. gemma4's full-layer
  share is smaller, so **its measured pool numbers and concurrency ceilings do not transfer**.
  Size this family's budget against `1/(1+3)`, not against gemma4's benchmarks.
- `max_chunk` needs a source. gemma4 wraps the model-neutral
  `crate::array::paged_prefill_chunk_size()` in a family constant, falling back to
  `GEMMA4_PREFILL_STEP_SIZE = 512` when the configured value is `<= 0`
  (`gemma4/model.rs:9187-9197`). Muse-Glimmer needs the same wrapper with its own fallback, and
  changing that constant is a **pool-sizing** change, not a perf knob.
- Mirror gemma4's reserved-blocks rule verbatim rather than re-deriving it: a sliding group
  **widens** to `max_admission_blocks.max(scheduler_width) + 1`, a full group stays at
  `max_admission_blocks` (`gemma4/model.rs:8169-8180`). Mirror the **reason** too, and note it
  is a *different* `+1` from the one in the admission formula above. **That** `+1` is the
  straddled window block and is already spent inside `max_admission_blocks` (161 =
  `div_ceil(2047 + 512, 16) + 1`; vLLM's `max_memory_usage_bytes` is `max_blocks *
  page_size_bytes` with nothing added on top). The reservation's `+1` is the group's
  **null-block sentinel** — gemma4's `null_block_bytes` term, one block per sliding group
  (`gemma4/model.rs:2148-2157`), which is why `required_bytes_for_width(1)` equals
  `minimum_pool_bytes` exactly (`:2182-2196`) and why gemma4's own test calls it "plus its null
  block" (`:12783-12793`). Two `+1`s, two blocks. Reading them as one invites deleting either,
  and both deletions bite: drop the seam's and every sliding group is under-provisioned by the
  straddled block, so the first prompt crossing a non-block-aligned window boundary allocates
  past its admission bound; drop the reservation's and the pool deadlocks at full occupancy with
  no null block for `remove_skipped_blocks` / `replace_block` to retire slots into. Pinned by
  `muse_glimmer/kv_cache.rs`'s
  `the_reservations_plus_one_is_the_null_block_not_the_straddled_window_block`.

#### Hybrid KV rules, each with the reason and where it is enforced today

| Rule | Reason | Enforced at |
| ---- | ------ | ----------- |
| **Never narrow a sliding block-table row.** The admission cap belongs to allocation accounting only | The row index **is** `absolute_position / block_size`. Blocks are recycled; rows are append-only. Narrowing the row breaks the index identity, which RoPE positions and the sliding mask both depend on | vLLM's `SlidingWindowSpec` deliberately has no `max_num_blocks_per_req` override (`kv_cache_interface.py:242`); our `prune_sliding_window_for` replaces in place |
| **Substitute a null-block sentinel; never compact.** | Compacting shifts every later row index. Absolute positions are load-bearing twice over — for RoPE on the 39 sliding layers and for the kernel's own window mask | `paged_kv_cache_adapter.rs:3662-3672` — `table.replace_block(index, null_block)` |
| **Retirement cutoff uses floor, not `div_ceil`.** | `first_live_block = (num_tokens - window) / block_size`. `div_ceil` would retire the block that still holds live in-window tokens | `paged_kv_cache_adapter.rs:3656-3657` (integer division) |
| **Flush pending pool writes before reclaiming.** | A freed block can be handed straight back out; an unflushed write then lands in someone else's block | `eval_pending_pool_writes()` at `paged_kv_cache_adapter.rs:3644`, before the free loop |
| **The window comes from the LAYERS.** A group with mixed kinds must yield **no** window | A window applied to a full layer silently discards everything older than it — fluent, wrong, no error | Structural for us: grouping keys on `attention_kind`, so a mixed group cannot exist. vLLM has to enforce it in code — `get_kv_cache_spec_sliding_window` returns `None` for a `FullAttentionSpec` *even when that spec carries a non-`None` `sliding_window` field* (`kv_cache_interface.py:970`) |
| **Never publish sliding blocks into the content-addressed prefix cache.** | A sliding block's contents depend on where the window was when it was written, not only on the token prefix that hashes to it | `gemma4/model.rs:490-494`: `if is_full { finalize_turn_keep_live_per_block(..) } else { finalize_turn_keep_live_no_prefix() }`; cold capture uses `full_adapter_mut()` only (`:499-510`) |
| **A hybrid prefix hit is joint, or it is refused.** Never a full-group-only hit | The groups must resume at one boundary or the sliding layers' KV describes a different prefix than the full layers' | `gemma4/model.rs:4884-4887` hard-errors when the sliding primed boundary differs from the full one; `:409` refuses group disagreement on the live continuation boundary |
| **NoPE is invisible to scheduling and to cache reuse.** Do not try to exploit position independence | Block hashes are **chained over the prefix**, so a cached block is bound to the offset it was computed at regardless of layer kind. Confirmed by exhaustion in vLLM: a word-boundary grep for rope/nope/rotary/theta across all of `vllm/v1/core/` and `kv_cache_interface.py` returns only two MLA byte-layout comments | — (this supersedes §2.1's "unusually friendly to cross-request prefix reuse", which is true of the *arithmetic* and **not** actionable in the cache) |
| **Keep `layer_rope_theta == 0` and `sliding_window == 0` in separate namespaces.** | They are different sentinels. `layer_rope_theta == 0` means NoPE. `sliding_window == 0` is read by our own Metal kernel and C++ validator as *"disable the sliding mask"*, i.e. **full causal attention** | `mlx_paged_ops.cpp:482` — "use 0 to disable the sliding mask"; `config.rs` refuses a zero window since `3c0c6859` |
| **"Full" and "NoPE" stay two independent per-layer facts.** | The coupling is Muse-Glimmer-specific, not structural. Cohere2-MoE has full layers that **keep** RoPE (`cohere2_moe.py:213` sets `force_rope`); Olmo3 selects different `rope_parameters` per `attn_type`. And NoPE does not mean position-independent: Llama 4 applies `attn_temperature_tuning` **only** on its NoPE layers, as a function of positions (`llama4.py:207`) | `config.rs` already keeps `layer_kinds` and `layer_rope_theta` as separate tables and asserts the coupling bidirectionally. **Keep the assertion, keep it labelled as this family's fact, and do not lift it into the seam** |

One correction to carry, from re-reading vLLM's current tree at `b369f10d5c`: "a divergent
per-group hit reconciles to **zero**" is too strong as a general statement. vLLM reconciles to
the greatest **common** boundary — a fixed-point loop drives `hit_length` down until every
group accepts it — and separately refuses a full-group-only *deeper* hit
(`kv_cache_coordinator.py:747`) rather than discarding the whole hit. Zero is only the
degenerate case. Our stance is stricter still and simpler: gemma4 takes one joint boundary and
**errors** on disagreement, and Muse-Glimmer should copy that, because "min" needs a
reconciliation loop we do not have. Record it as *joint or refuse*, and know that the strict
version is our choice, not an inherited necessity.

Also on the record: vLLM imposes **no** limit on the number of full-attention groups
(`HybridKVCacheCoordinator` asserts only a lower bound of 2 groups), so gemma4's ">1 full
group" refusal is our own shortcut. It is accidentally protective — with two *distinct* full
specs, `find_longest_cache_hit` truncates blocks for `attention_groups[0]` only
(`kv_cache_coordinator.py:818`), so a second full group can return blocks past a reconciled hit
of 0. Muse-Glimmer's uniform geometry means one full group, so this cannot bite us; if the
refusal is ever relaxed, the truncation must loop over **all** full groups first.

#### The window-blind cache-hit prefill — MEASURED, and what Muse-Glimmer does about it

Status: **confirmed by reading the code, then reproduced twice and quantified on real
weights.** Earlier revisions of this section said "the end-to-end quality impact is
unquantified" and prescribed the wrong fix. Both are corrected below; the numbers replace the
hedge.

**STATUS: FIXED ON THIS BRANCH.** Everything from here to the end of this subsection is the
diagnosis, kept because M1 needs to know what the shape of the bug was. The tree no longer
behaves this way:

| Commit | What it landed |
| ------ | -------------- |
| `179e826d` | The regression test: a sliding window must survive a cache-hit prefill chunk |
| `eb5713e3` | The fix: the window travels through every cache-hit prefill route |
| `1904b138` | The "sliding adapter x attention numerics" cell, occupied |
| `7a898db4` | One dense cache-hit implementation, made testable |
| `f28e99f4` | The window is taken from the gather that produced the K/V, and a mask is asked for only when the window can bite |
| `c0faba4a` | The adapter refuses a windowed group on a window-blind dense read, for every model. Weaker than its first description here: `read_kv_range` refuses by WIDTH, not outright — see the table below |
| this round | The window is bounded at `i32::MAX` at config load, at spec derivation and at dispatch admission, and `PagedWindowSlot` stores the `i32` rather than casting a `u32`. Loud for the kernel route (already fail-closed in the C++ validator, so: diagnostics), load-bearing for the mask route (a negative width was an all-`false` mask, answered silently with the uniform mean of every key) |

**Copy this shape, do not re-derive it.** `CacheHitPrefillPlan.sliding_window` is a
`DenseAttentionWindow` (`transformer/paged_kv_cache_adapter.rs`) with no public constructor, so a
literal `0` is untypeable at every dense call site; `dense_cache_hit_attention`
(`gemma4/attention.rs`) is the single dense implementation and takes that type as a required
parameter; and `gather_kv_for_dense_cache_hit_prefill` /
`read_kv_range_for_dense_attention` return the window **with** the K/V, while the window-blind
`gather_kv_for_prefill_sdpa` / `read_kv_range` `Err` for a windowed group.

**That last clause was overstated and is corrected here**, because M1 is meant to lean on it. The
two window-blind readers refuse with different strengths, verified by reproduction:

| reader | refuses | does NOT refuse |
| ------ | ------- | --------------- |
| `gather_kv_for_prefill_sdpa` | **any** sliding group, unconditionally (`if self.sliding_window != 0 { Err }`) | — |
| `read_kv_range` | a read **wider than the window** (`num_tokens > sliding_window`) | a read whose length is `<= window` no matter how old it is; `start_pos` is not consulted |

So the honest statement is: **M1 cannot obtain placeholder-laden dense K/V for a route whose
context exceeds the window** — which is the only regime where being window-blind is wrong, since
below it nothing is out of range and nothing has been retired. It is *not* true that
`read_kv_range` refuses a sliding group outright. Measured on a window-16 / block-8 adapter with
48 recorded tokens: after `prune_sliding_window_for` replaced table indices 0..3 with the reserved
null block, `read_kv_range(0, 0, 16)` returned **`Ok`** with all-zero K/V where `1001..1016` had
been written — i.e. the never-written null block, whose contents are formally **undefined**
(`layer_kv_pool.rs:499`: `StorageModePrivate`, *"not zeroed … returns whatever the driver handed
out"*). Today's all-zero readback is a driver accident, not a guarantee, in either direction.

No caller bleeds that data today (`new_sliding` has one production callsite; the only sliding
`read_kv_range` caller, gemma4's cold-tier capture, declines out-of-window ranges itself), so this
is a prospective hole plus a documentation defect rather than a live bug. If a live-tail
(`start_pos >= recorded - window`) arm lands in the adapter, the row above tightens and this note
should say so; do not upgrade the wording ahead of the code.

```
prefill body chunk 1..n           (cached_prefix_len = absolute_position > 0)   [AS OF eb5713e3^]
  model builds a real sliding mask     gemma4/model.rs:5177  create_sliding_mask(...)
  threads it in as `mask`              gemma4/model.rs:5197  kind.is_sliding() => Some
  forward_paged                        gemma4/attention.rs   explicit_prefill_mask
    cached_prefix_len == 0  -> mask IS consumed
    cached_prefix_len != 0  -> forward_paged_cache_hit_prefill(x, q, adapter,
                                 paged_idx, cached_prefix_len)
                               ^ signature has NO mask parameter
                                 => the mask is structurally dropped
      of its four sub-paths, only ForceLegacy passed the window:
        PagedPoolSdpa   -> gather_kv_for_prefill_sdpa + causal SDPA, no mask   <-- WAS DEFAULT
        PagedVarlen     -> literal 0 in the kernel's sliding_window slot
        HostRead        -> read_kv_range(0, total_ctx) + create_causal_mask(.., None)
        PagedLegacy     -> passes self.sliding_window                (diagnostic only)
```

**The default route is the dense SDPA one, not the varlen literal.** This is the single most
important correction, because an earlier revision led with the `0` literal and a reader would
have fixed only that. Traced on the real checkpoint with
`MLX_NODE_LOG=mlx_core::inference=info` at prompt 1475: `planned_path="paged_pool_sdpa"` for
**both** KV groups including the sliding one, `effective_sdpa_dtype=None` (that `None` **is**
the sliding group), graph constructed and not rejected. `Auto`'s branch order tests
`estimated_sdpa_bytes <= budget` first — 171.6 MiB against `(headroom - 2 GiB) * 0.9` — so
varlen is reached only under real memory pressure (live headroom below roughly 2.2 GiB), and
`HostRead` only when the graph backend is unavailable. Fixing the literal alone would have left
every default-configuration turn just as wrong **while turning the new regression test green**,
which is the worst outcome available.

Measured impact, two independent measurements:

| Measurement | Result |
| ----------- | ------ |
| Kernel A/B, one process, one physical pool, sliding adapter window 1024, 4096 tokens written in 512-token chunks with prune after each, target chunk `[3584,4096)` | max abs delta vs the windowed reference: **0.124512** with window slot `0`, **0.000977** (1 bf16 ULP) with window slot `1024`. `rms(correct output) = 0.035562`, so the error is **3.5x the signal's own RMS** and 128x the correct route's residual — from one argument |
| Real gemma-4-12b-it greedy continuation, flat path as ground truth, temperature 0. **Two runs, not interchangeable** | `MLX_GEMMA4_PAGED_PREFILL_ROUTE=sdpa` (the route `Auto` selects) over 9 lengths: **6 of 9** mismatch — matches at 1205 / 1307 / **2771**, mismatches at 1374 / 1424 / 1475 / 1540 / 2066 / 4008. Last match 1307, **first mismatch 1374**, untested between. Route unset (`Auto`) over 10 lengths: **4 of 10** mismatch, first mismatch **1475**; `Auto` was never run at 1374, so its own bisect interval is 1205-1475. Divergences are fluent and coherent (first differing char at ~token 13), i.e. a silently different model, which is why it went unnoticed for so long. Note 2771 matching on **every** route while the defect was live: divergence is not monotonic in prompt length, so a single-length parity gate can be green for the wrong reason |
| Null-block regime | varlen's output matches "full causal over the whole 4096-token context with the pruned range treated as zero K/V" to **0.000244**, below bf16 ULP — so the window-0 kernel verifiably dereferences 2560 never-written slots per row. Those slots read back `max\|K\| = max\|V\| = 0` on this machine's driver. **Do not rely on that in either direction**: `layer_kv_pool.rs:499` says the pool is `StorageModePrivate` and **not zeroed**, and the probe used `Q = 0`, so it bounds the V bytes and says nothing about a logit blow-up from garbage K under real non-zero Q |

**Two damage regimes, with corrected thresholds.** The earlier ~514 / ~1026 figures came from
gemma4's serde default `sliding_window() -> 512`, which **never applies**: every gemma-4
`config.json` on disk sets the key explicitly (12b-it / 26b-a4b / 31b-it = 1024; e2b = 512).

| Regime | Onset (gemma-4-12b-it, window 1024, chunk 512) | Onset (Muse-Glimmer, window 2048, chunk 512) |
| ------ | --- | --- |
| **Over-attention**: sliding layers attend real out-of-window tokens. First body chunk with `cached_prefix > 0` and `total_ctx > window` | ~1026 prompt tokens (structural). First greedy-token flip **measured** on the `sdpa` route between 1307 (match) and **1374** (mismatch); on `Auto` between 1205 and **1475**. The interval is what was measured — no length between the two was tested | ~2050 prompt tokens |
| **Undefined read**: the block table holds the reserved null block, and the kernel dereferences K/V slots that were never written. Needs a chunk end >= `window + block_size`, and chunk ends are multiples of 512 | ~1538 prompt tokens | ~2562 prompt tokens |

Muse-Glimmer's exposure is strictly worse in shape even though its onset is later: **39 of 52
layers (75%)** are sliding versus gemma4's 40 of 48, and the wider window means each affected
row over-attends a larger absolute span.

**The fix shape — corrected.** An earlier revision said *"a windowed adapter reaching a path
that cannot carry its window is an `Err`, not a silent `0`. Fail closed … three of the four
sub-paths have no window concept at all."* Both halves are wrong, and the second is dangerous:

- **All four sub-paths CAN express the window**, so there is nothing to refuse. The varlen
  kernel is already per-query-row bottom-right aligned
  (`paged_attention.metal:1860` `effective_context_len = context_len - q_len + q_pos_in_seq + 1`;
  `sliding_lower` derives from it), so one `i32` fixes a whole multi-token chunk with no
  per-row work by the caller. And `create_causal_mask(seq_len, offset, window_size)`
  (`array/mask.rs:33-74`) **already has the window parameter** — `linds >= rinds && linds <
  rinds + window`, bool `true = keep` — and the host-read call site was passing `None` into it.
  "No window concept at all" was wrong for two of the three: the concept, the argument and the
  builder all existed. Refusing instead of threading would turn every gemma4 prompt over ~1026
  tokens into an error.
- **An `Err` raised inside a gather does NOT fail closed.** Each gather failure on that path is
  downgraded to `tracing::warn!` and chains to the next route, terminating at an
  *unconditional* host-read. So a refusal below `forward_paged_cache_hit_prefill` is swallowed
  and re-routed to another window-blind path — the opposite of fail-closed. Any refusal has to
  be raised **at or above** that function.

So the shape was — and, as of `eb5713e3` + `f28e99f4`, is: **thread the window, at the single
choke point, unconditionally.** The window
is a property of the layer and must travel with **every** dispatch — vLLM's rule, where
`sliding_window` lives on the `AttentionImpl` and is passed to every `flash_attn_varlen_func`
call. Verified at `b369f10d5c` and again at `2ac1f683f1`: **all seven**
`flash_attn_varlen_func` dispatch sites in `vllm/v1/attention/backends/flash_attn.py` pass
`window_size=` — `:1134` (inside `forward`, the load-bearing one: it carries the mixed
prefill+decode batch), `:1250` / `:1340` / `:1372` (inside `_forward_with_dcp`, which starts at
`:1215`), `:1447` (inside `_forward_encoder_attention`, `:1391`, three lines under `causal=False`)
and `:1752` / `:1780` (cascade). An earlier revision of this doc labelled `:1340/:1372/:1447` as
"context / query / decode"; all three labels were wrong — two are decode-context-parallel-only
and the third is bidirectional encoder attention — and the unified dispatch was uncited. A
per-route argument list is where a window gets lost.

Three things must not move, each because moving it silently undoes the fix:

1. **`prune_sliding_all` runs AFTER the layer loop** (`gemma4/model.rs:4913-4940`:
   `record_tokens_all` -> `run_paged_prefill_layer_loop` -> `eval` -> `prune_sliding_all`). For
   a chunk ending at `E` the cutoff is `E - window`, and the next chunk's earliest query row
   needs from `E + 1 - window > cutoff`, so every position any row needs is still live.
   Reordering prune before the layer loop converts a correct window into **under**-attention.
   This is our version of vLLM's `sliding_window - 1 + max_in_flight_tokens` reservation
   (`kv_cache_interface.py:618-622`).
2. **Do not plumb the existing `sliding_mask`** from `run_paged_prefill_layer_loop` into the
   cache-hit branch. `create_sliding_mask` builds it for the TRIMMED rotating view — width
   `seq_len + min(cache_offset, window)` — while the paged K/V is `cached_prefix_len + seq_len`
   wide. `trim_mask` errors on a short mask, so it fails closed rather than silently, but it is
   still the wrong value. Build the mask inside the helper from `cached_prefix_len`. The flat
   path keeps using the trimmed one; leave the flat path alone.
3. **Do not reinstate a sliding guard on the prefill-side gather dtype.** Once the dense SDPA
   route carries a window mask the gather is legal for a sliding group, and a `None` there only
   doubles `dtype_bytes` 2->4 and biases routing on a byte estimate rather than on correctness.
   The same guard on the **decode** side IS load-bearing — it forces the paged kernel, which is
   why decode is correct and needs no fix — so the accessor wants splitting, not deleting.

**`MLX_GEMMA4_PAGED_PREFILL_ROUTE=legacy` is not a mitigation.** It is the one route that
passed the window, so it reads like a workaround. Measured: it diverges from flat at 2066
prompt tokens, 3/3 independent processes, identical wrong text, and produces the *same* wrong
continuation as `Auto`/`sdpa` there — it has its own defect at short-final-chunk geometry (a
17-token chunk at `cached_prefix 2048`). Diagnostic reference only.

**What landed for Muse-Glimmer, and the honest split.** Muse-Glimmer has no forward pass, so it
cannot inherit the bug today — `new_sliding` / `SlidingPaged` appear nowhere under
`models/muse_glimmer/`. It would inherit it by construction the moment M1 wires a sliding
adapter. The contract therefore lands in the KV-cache seam first, in `muse_glimmer/kv_cache.rs`:

| Item | What it is |
| ---- | ---------- |
| `WindowCarrier` | How a route can express a window: `KernelArgument` (the kernel's `sliding_window: i32`), `ExplicitMask` (`create_causal_mask`'s `window_size`), `Unsupported`. Supplied by the dispatch site, because capability is a fact the route knows and the group does not. A fifth route means a new variant, which is a **compile error** at the admitting match rather than a silently window-blind path |
| `PagedWindowSlot` | The window argument for one admitted dispatch. Private fields, no `Default`, no public constructor — the only way to get one derives the window from the group's own `AttentionKind`, so **there is no parameter for a literal `0` to be written into**. Its two accessors are carrier-scoped and both return `Option`: `kernel_slot()` answers only on kernel routes, `mask_window()` only on mask routes, and each returns `None` — never a plausible `0` — when asked the other route's question. The window is stored as the **`i32`** both consumers take, range-checked once at admission, so **no cast survives inside the type**: storing the spec's `u32` and casting in the accessors is what let a config window in `[2^31, 2^32-1]` present itself as a NEGATIVE window (`3000000000` -> `-1294967296`) |
| `paged_window_for_kind` / `paged_window_for_group` | Admit one dispatch, or `Err`. Refuses **three** things: a windowed group on an `Unsupported` route; a `SlidingWindow { sliding_window: 0 }` payload, a third time (config load and spec derivation being the first two) because by the time a 0 reaches a kernel slot it is indistinguishable from "disable the mask"; and a window **past `i32::MAX`**, because both consumers take an `i32` and the mask route turns a negative width into an all-`false` mask that the fused SDPA answers with the uniform mean of every key — no NaN, no error. The group-level form names the **39 layers** in its error; "group 1" understates the blast radius |
| `admit_paged_dispatch_plan` | The per-turn choke point. Admits every group against the routes that turn selected, before any launches. `carriers` is indexed by `group_id`, so a plan that forgets a group is an arity error and a permuted group list is refused — pairing the full group's carrier with the sliding group is the same defect with a different cause |

A full-attention group is admitted on **every** route, including `Unsupported`, and asks for no
mask. That is deliberate over-strictness avoidance: refusing it would refuse a legal global
layer, and `mask_window() == None` for a full group is itself load-bearing — the mask-bearing
SDPA kernel has a different bf16 reduction order from the `_causal` one, so materializing an
all-true mask for the 13 global layers would move their numerics with no correctness gain, and
a paged-vs-flat parity gate would then fail for a reason unrelated to windows.

**Enforced today** (model-free, in CI): a windowed group paired with a window-blind route is an
`Err`; no admitted sliding dispatch can present a window of `0` through either accessor
(enumerated over all three carriers, not spot-checked); a full group keeps the unmasked causal
path; the window tracks the config rather than a constant.

**Now enforced by the adapter, for every model** (`c0faba4a`): `gather_kv_for_prefill_sdpa` `Err`s
for a windowed group unconditionally, `read_kv_range` `Err`s for a read **wider than the window**,
and the only dense readers that serve one return a `DenseAttentionWindow` alongside the K/V.
`DenseAttentionWindow` has no public constructor, so a Muse-Glimmer forward pass cannot obtain
placeholder-laden dense K/V **for a context wider than the window** without holding it, in any
file, including a shared one this family's source tripwire never reads. That closes the "wiring
lives outside `models/muse_glimmer/`" hole in the tripwire below for the regime that matters —
below the window there are no placeholders to be laden with. See the corrected table in
"Copy this shape, do not re-derive it" for what each reader does and does not refuse; the earlier
"`read_kv_range` `Err`s for a windowed group" was too strong and `read_kv_range(0, 0, window)` on
a pruned adapter returns null-block bytes with `Ok`.

**Only M1 can enforce** — stated as a requirement, not a hope, because the type system cannot
prove it: **that every dispatch actually asks.** A forward pass can always call the FFI with a
hand-written argument. The testable form that exists today is the source tripwire
`wiring_a_sliding_adapter_without_this_seam_trips_the_tripwire`: the moment paged-attention
wiring (`new_sliding`, `mlx_paged_attention*`, `gather_kv_for_prefill*`, `read_kv_range`,
`scaled_dot_product_attention_causal`, `PagedKVCacheAdapter`) appears anywhere under
`models/muse_glimmer/` — the scan is **recursive** as of `34121b01`, so an
`attention/paged.rs` subdirectory is opened, and the file count is pinned from both sides so
growing the family forces a look at the test — that file must at least NAME `PagedWindowSlot`. It
is **vacuously true today, and that is the point — it is armed, not satisfied.**

Three things it does **not** do, named here because overstating it is the failure mode it exists
to prevent, and because an earlier revision of this paragraph said "appears in any
`models/muse_glimmer/*.rs`", which was literally false for a subdirectory:

| Hole | Covered by |
| ---- | ---------- |
| **Textual.** A bare `use super::kv_cache::PagedWindowSlot;` beside a hand-written FFI call satisfies it | Nothing. Naming the seam is the floor |
| **Per-file, not per-call.** One admitted dispatch covers for a second hand-written one in the same file | Nothing. This is the "only M1 can enforce" part |
| **This family's directory only.** Wiring in `transformer/block.rs` or a new generic paged forward trips nothing | The adapter refusal (`c0faba4a`) — a shared route cannot serve a windowed group window-blind, whoever calls it |

It does not prove every dispatch is admitted; it does make "wire a sliding adapter and forget the
contract entirely" impossible to do green, which is exactly what happened in gemma4, where no test
occupied the "sliding adapter x attention numerics" cell at all.

So M1 owes three things, each with its reason:

1. **Every paged dispatch takes its `sliding_window` from `admit_paged_dispatch_plan`.** Reason:
   the window is the layer's property and a per-route argument list is where it gets lost.
2. **Route selection reports its real `WindowCarrier`.** Reason: a route that gains the ability
   to carry a window should stop being refused automatically, and a route that loses it should
   start being refused automatically. Hard-coding `KernelArgument` to silence the seam
   reproduces the defect with the guard still in the file.
3. **A numerics test through a real attention call, on a sliding adapter, at a cached prefix.**
   Reason: this is the cell no gemma4 test occupied. The model-free kernel A/B separates a
   correct fix (<= 1e-3, bf16 ULP) from the broken behaviour (0.1245) by 128x and needs no
   weights. Add it alongside `muse_glimmer_paged_vs_flat_parity` as a **sweep** past this
   family's over-attention onset — onset+ε, onset+2 chunks, onset+4 chunks — not one long
   prompt. Reason, and it was measured: in gemma4 every route matched flat at 2771 tokens while
   the defect was live, with mismatches on both sides of it. A single length past the onset can
   be green for the wrong reason exactly as a short one can.

**Comments elsewhere that asserted this invariant and did not enforce it.** Recorded because the
dropped mask survived behind exactly these sentences, and because a reader who trusts them will
not look. **Each bullet now carries its current status** — do not grep for a sentence that was
deleted:

- `paged_kv_cache_adapter.rs` `null_block` field doc — "the paged attention mask guarantees
  placeholder positions are outside the live window before their original physical block is
  reclaimed". This is the **stated safety invariant of the whole prune design**, and it was
  false on three of the four cache-hit prefill routes. **Status: text unchanged, invariant now
  restored by the caller** (`eb5713e3`) and by the adapter refusing a window-blind dense read at
  all (`c0faba4a`). Still a property of the caller, not of the kernel.
- `prune_sliding_window_for`'s doc — "masks every placeholder position before dereferencing its
  K/V", stated as a property **of the kernel** rather than of the caller, which is precisely how
  the caller-side hole stayed invisible. The kernel does this only when told the window.
  **Status: text unchanged, same restoration as above.** Read it as a requirement on whoever
  dispatches, which is what `DenseAttentionWindow` now enforces in the type system.
- `prefill_sdpa_cache_dtype()`'s "Sliding groups must remain on the paged kernel, which applies
  the window before touching those entries" — was measured false twice: traced to
  `paged_pool_sdpa` on real weights, and `gather_kv_for_prefill_sdpa` called directly on a
  sliding adapter succeeded and handed back a dense K/V with the retired null placeholders
  materialized in it. **Status: that sentence was DELETED by `eb5713e3`** — do not grep for it.
  The doc there now says the opposite of what it used to and points at the gather refusal
  (`c0faba4a`) as the invariant that actually exists. On **decode** the accessor is still
  load-bearing exactly as described further up this section.
- `paged_attention.metal`'s "zero contribution to softmax / V reduction" for a masked position.
  The **softmax** half is enforced (`stored_logit = -INFINITY`, and `exp(-INF - qk_max)` is
  exactly `0.0f`). The **V** half is not: both non-striped V loops zero lanes only past the
  *causal* cutoff, never below `sliding_lower`; only the grouped striped kernel checks both. So
  `0.0f * NaN = NaN` would poison a whole head from one stray byte, and gemma4 sliding **decode**
  runs the generic kernel, not the guarded striped one. Measured benign today (`max|K| = max|V|
  = 0`) — which is driver luck, not a contract. **Status: STILL OPEN**, tracked as the P2
  hardening below; the fixes above stop the mask from being dropped but do not make the kernel's
  V half honour `sliding_lower`.
- `gemma4/decoder_layer.rs`'s "`mask` is normally None for global layers … which `forward_paged`
  applies in the fresh-prefill branch". The arm it sits in handles `SlidingPaged | GlobalPaged`
  in ONE match; for sliding layers `mask` is the sliding mask, not `None`, and "`forward_paged`
  applies it" held only in the `cached_prefix_len == 0` branch. The comment reads as a complete
  account of what happens to `mask` and omits the only case that matters. **Status: STILL
  WRONG** — the code around it is fixed, the comment is not. It is the one bullet here that is
  still an active trap.

Two cheap hardenings, **P2 and separate**, because the undefined read measured benign today:
add the `value_token >= sliding_lower` guard to the two non-striped V loops (mirroring the
grouped striped kernel, and correcting the comment to claim only the softmax half), and/or
zero the pool once in `LayerKVPool::new` with a blit `fill_buffer` at construction —
`StorageModePrivate` is not CPU-writable — then correct the "not zeroed" doc. That is vLLM's
`torch.zeros` (`gpu_model_runner.py:7456-7458`): one memset at startup, zero steady-state cost.
It converts this whole class of missed mask from **undefined** to **wrong-but-bounded**, which
is what makes the `null_block` doc's guarantee survive the next bug. Both need a metallib
rebuild (`PATH=/usr/bin:$PATH SDKROOT=$(xcrun --show-sdk-path) yarn build:native`, never plain
`cargo build`).


#### Opt-in wiring: three lines, and two ways to be silently serial

The trait is opt-in. Three easily-forgotten lines, all verified against gemma4:

| # | Line | Where gemma4 has it |
| - | ---- | ------------------- |
| 1 | `pub(crate) type MuseGlimmerSchedulerState = HybridSchedulerState<MuseGlimmerInner>;` | `gemma4/scheduler.rs:62` |
| 2 | `MuseGlimmerSchedulerState::new(inner)?` at model-thread spawn | `gemma4/persistence.rs:2899` |
| 3 | `\|state, receiver\| state.drive(receiver)` as the loop body | `gemma4/persistence.rs:2912` |

And there is **no compile-time link between the native trait and the TS server**. The three
`#[napi]` methods — `has_block_paged_cache`, `max_concurrent_sequences`, `scheduler_stats`
(`gemma4/model.rs:7245/7250/7259`) — are **optional** on the TS side:
`hasBlockPagedCache?()` and `maxConcurrentSequences?()` at `packages/lm/src/chat-session.ts:476/486`.
`concurrentDispatchCapacity` (`packages/server/src/registry.ts:50-55`) returns `1` when either
getter is absent or reports `< 2`. Omit them and the model stays serial with **no diagnostic
anywhere** — no warning, no error, just single-dispatch throughput. Add all three in the same
commit as the trait impl.

#### The gate, and why its fourth assertion exists

For a **hybrid** the gate is a TypeScript E2E, not a Rust integration test. gemma4's lives at
`__test__/models/gemma4-concurrency-e2e.test.ts`, `describe.skip` unless its model-path env var
is present:

| # | Assertion | Line |
| - | --------- | ---: |
| a | `maxConcurrentSequences() >= 2` (and `hasBlockPagedCache() === true`) | 98-99 |
| b | `rawText` identical, serial replay vs two concurrent starts **and** two concurrent continuations | 112-121 |
| c | `maxBatchOccupancy >= 2` **and** some histogram bucket with `occupancy >= 2 && steps > 0` | 124-125 |
| d | `fusedGreedyEpilogueSteps > 0` | 126 |

**(d) is the only non-vacuity check.** (a)-(c) all pass under a scalar-loop fake: a backend
that reports capacity, runs rows one at a time and returns per-row logits satisfies capacity,
parity and occupancy while never executing the batched `[N, 1, vocab]` epilogue. Only (d)
proves the production fused path ran. The Rust-side wording in
`tests/concurrent_batched_parity.rs:290-341` says so directly — the mixed-penalty wave
"deliberately proves scalar fallback", and a separate penalty-free N=2 wave exists to engage
the real epilogue.

Two corrections to the received wisdom, both checked: the `tests/<family>_concurrent_batched_parity.rs`
files exist for lfm2, qwen3_5 and qwen3_5_moe and assert only (a)-(c); only qwen3's
`tests/concurrent_batched_parity.rs` asserts (d). **gemma4 — the only existing hybrid — has no
such Rust file at all**; its Rust suites assert `max_concurrent_sequences() >= 2` alone
(`gemma4_assistant.rs:103`, `gemma4_dspark.rs:102`) and the full four-assertion gate is the TS
E2E above. Follow gemma4, and remember that TS E2E legs are **opt-in on PRs** — without the
`model-e2e` label the leg is skipped and the first real run lands on `main`.

#### Cold tier (M4) is two allowlists plus one scheduler opt-out

`COLD_RESTORE_FAMILIES` (`crates/mlx-core/src/cold_tier.rs:463-470`) and
`COLD_TIER_RESTORE_FAMILIES` (`packages/agent/src/cold-tier.ts:73-80`) are drift-tested against
each other; both need `"muse_glimmer"`, and the native list's own doc comment says widening it
is authorized by exactly one thing — the family's restart-parity gate passing on real weights
with `hits > 0` and `corruptions == 0`.

Separately, a hybrid should ship `scheduler_has_cold_tier() -> false` at first, as gemma4 does
(`gemma4/scheduler.rs:168-173`): its full/sliding sidecar is a **joint commit record**, so
ordinary prefix admission may restore it, but scheduler **preemption** must drop every group
and recompute rather than offloading only the full group. That override is what downgrades
`PreemptionMode::Ssd` to `Recompute` (`hybrid_scheduler.rs:1963-1965`) and keeps the
`WaitingForSsd` lane out of play.

#### Two claims already on the record that are wrong or overstated

1. **`CLAUDE.md:59` was stale, and was stale the moment it was written — FIXED by `17dc2bdb`,
   which this spec no longer needs to carry.** It now reads "Concurrent chat inference is
   per-family, not global…", names the eligible families and both env vars, and points at
   `docs/concurrent-inference.md`. The history, for anyone reading a pre-`17dc2bdb` checkout: it
   said "Chat
   inference is serialized per model: one `"mlx-model"` OS thread per loaded model runs one
   whole turn at a time, and the server adds a per-model FIFO mutex." `37f1d68e` **added that
   line and contradicted it in the same commit**: `docs/concurrent-inference.md:22-31` (also
   added by `37f1d68e`) says different sessions on one eligible Qwen3 / LFM2 / Qwen3.5
   dense-or-MoE / **Gemma4** paged model may overlap, that the server admits up to the native
   scheduler's physical sequence capacity, and that the model thread advances them together.
   `docs/concurrent-inference.md` is the accurate one.
2. **"Gemma4 cold persistence is disabled" (37f1d68e's own commit message) is overstated.**
   `gemma4` **is** in `COLD_RESTORE_FAMILIES` (`cold_tier.rs:463-470`) and in the TS list, and
   its ordinary cold path is live: `resolve_persist_cold("gemma4", ..)` at
   `gemma4/persistence.rs:2436`, `try_install_cold_sidecar` at `gemma4/model.rs:2581` and
   `:3281`. What is actually disabled is the **scheduler's** SSD escalation —
   `scheduler_has_cold_tier() -> false` (`gemma4/scheduler.rs:168-173`), which forces
   `PreemptionMode::Ssd` down to `Recompute` and keeps the `WaitingForSsd` lane unused. Do not
   repeat the commit message's phrasing; state the narrower fact.

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
tokens = 1.745 GB, 39 sliding layers = **102.9 MB** at `max_chunk 512`. (Not 81.8 MB: that is
`div_ceil(2048, 16) = 128` blocks, which omits the in-flight chunk. Admission is
`div_ceil(window - 1 + max_chunk, block) + 1 = 161` blocks — §8's sizing table is the authority,
and the figure **rises with `max_chunk`**: 123.3 MB at 1024, 164.2 MB at 2048. This section is
where someone sizes `paged_cache_memory_mb`, and §8 also says that must be set explicitly
because oversizing tanks long-context decode ~10x, so understating it here is the expensive
direction.) The pressure is elsewhere:
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
