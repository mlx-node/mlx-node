# CLEF decisions

CLEF and CLEF Flash are text decision models built on the dense Qwen3.5/Qwen3.8
backbone. They answer `noul` (yes/no probability), `choice`, and `score` questions
in one backbone pass. They are exposed separately from chat models.

## Checkpoints

A checkpoint directory must contain the Qwen config, tokenizer, backbone weights,
`joint_head_config.json`, and `joint_head.safetensors`. Model discovery identifies
CLEF from both head assets. An incomplete head is an error.

The regular model converter copies both head assets unchanged. Backbone
quantization is supported by the existing Qwen loader. `lm_head` may stay
packed: the decide path never runs it as a projection and dequantizes it on
demand for the joint head's option-embedding gather. Tied output embeddings
are unsupported.

A decision-model GGUF (`general.architecture: clef`) loads directly through
`ClefModel.load`. Conversion splits the `dec.*` / `decision.*` tensors into
`joint_head.safetensors` and synthesizes `joint_head_config.json` beside the
backbone `model.safetensors`, cached under the shared native-GGUF staging
directory. The joint head itself must sit in affine ggml blocks (the
published files keep every head weight in Q8_0); a K-quant head weight is
rejected at conversion.

### Residency

A decide is one prefill forward, so each backbone weight is read once at
`M = prompt length`. The loader therefore rewrites the quantized backbone
projections at load time instead of running the checkpoint's packed matmul
kernels:

| Mode   | Storage after load                                       | GEMM route                             |
| ------ | -------------------------------------------------------- | -------------------------------------- |
| `sym8` | int8 `[N,K]` weight + f32 `[N]` per-output-channel scale | int8 W8A8 (per-token activation quant) |
| `bf16` | dense bf16                                               | MLX bf16 GEMM                          |

`sym8` is taken for a quantized checkpoint on a gen >= 17 GPU (M5 and later)
and `bf16` otherwise (older GPU, or an unquantized checkpoint, which is left
as stored). Measured on an M5 Max at the 9B and 27B CLEF projection shapes,
the sym8 GEMM alone is 1.3-1.8x faster than the dense bf16 GEMM at every `M`
from 128 to 1024, while every weight-only packed route (affine qmm, the tiled
`a8g64@t64` / `q4k@t64` kernels, per-group W8A8) is slower than dense bf16
from `M >= 256`. The cost is the per-token int8 activation quantization,
which the rotation below keeps inside the acceptance gate.

End to end on the published GGUFs (`ggml-org/Clef-GGUF` and
`Clef-Flash-GGUF`, Q8_0, the six acceptance inputs, M5 Max, one process per
run), against the bf16 real-weight results in
[validation](validation/clef-real-weights.md):

| Checkpoint    | Residency | Sum of 6 decides | Resident MLX memory | Max abs probability diff |
| ------------- | --------- | ---------------: | ------------------: | -----------------------: |
| Flash 9B Q8_0 | `bf16`    |           1.47 s |             19.2 GB |                   0.0012 |
| Flash 9B Q8_0 | `sym8`    |           1.14 s |              9.3 GB |                   0.0071 |
| CLEF 27B Q8_0 | `bf16`    |           8.60 s |             61.8 GB |                   0.0049 |
| CLEF 27B Q8_0 | `sym8`    |           6.78 s |             27.3 GB |                   0.0047 |

Every modal decision matched in every run. Without the rotation sym8 is
another ~10% faster but the max diff rises to 0.038 (9B) / 0.015 (27B), over
the 0.01 gate, so the rotation is always on. The Q4_K_M GGUFs are dominated
by their weight quantization (0.08 on Flash even under `bf16`).

The requantization uses the same routine as `mlx convert --q-mode sym8` and
follows its ladder: 2-D weights with `K % 16 == 0`, excluding `in_proj_ba`
(and the split `in_proj_b` / `in_proj_a` halves), which stays dense bf16.
An MLP is decided as one unit: a dense sibling in the checkpoint or one
ineligible member sends all three projections to bf16, because the MLP
installs all-or-nothing. A projection the checkpoint already stores as sym8
is used as is — unrotated, the numerics of the `sym8` convert format itself;
the rotation below applies to load-time requantization only, and the
measured gate results are for the published GGUFs. The embedding and
`lm_head` are never rewritten in either mode: the lookup and the joint
head's row gather read packed rows directly.

Each requantized projection whose input width `K` is a multiple of 1024 gets
a QuaRot-style online Hadamard rotation: the dense weight is rotated along
`K` in f32 (`W' = W · H_s`, `H_s` a signed 1024-block Walsh-Hadamard with a
fixed sign vector derived from `K` alone) before the int8 quantizer, and the
built projection applies the same transform to its activation before the
W8A8 kernels, so `(x · H_s)(W · H_s)^T = x W^T` while the per-token int8
activation quant sees outlier-spread rows. Sign vectors are deterministic
per `K` (bit-identical across processes) and shared by every projection of
one width. Projections with `K % 1024 != 0` requantize without rotation; the
loader summary logs both counts.

## TypeScript

```ts
import { ClefModel } from '@mlx-node/lm';

const model = await ClefModel.load('/absolute/path/to/clef-flash');
const result = await model.decide({
  state: 'The customer received a damaged parcel and requests a replacement.',
  questions: {
    replace: { type: 'noul', instructions: 'Does the customer want a replacement?' },
    priority: {
      type: 'choice',
      instructions: 'Choose the support priority.',
      criteria: { normal: 'Routine support request', urgent: 'Immediate safety concern' },
    },
    sentiment: {
      type: 'score',
      instructions: 'Rate the sentiment.',
      criteria: ['Negative', 'Neutral', 'Positive'],
    },
  },
});
console.log(result.answers);
```

`decide(request, { signal })` accepts an `AbortSignal`. Cancellation is checked
between backbone layers and before and after the joint head. Work already running
inside a Metal operation must finish before cancellation takes effect.

`decideRaw(json, { signal })` preserves question order from raw JSON, including
integer-looking question IDs that JavaScript object enumeration would reorder.
State and descriptions can contain JSON values. Option names are sorted using
Unicode ordering, and structured descriptions use compact, sorted JSON.

## HTTP and TypeSafe

Put the checkpoint in a child directory of your models directory:

```sh
mlx serve --models-dir /absolute/path/to/models --model clef-flash --port 8080
```

Send `POST /v1/systemone` with `model`, `state`, and `questions`. The response
contains `model`, `answers`, and `usage`, with `output_tokens: 0`. The
`x-typesafe-request-id` response header identifies the request. Authentication
uses the server's existing bearer-token configuration.

```ts
import { TypeSafeClient } from '@typesafe-ai/sdk';

const client = new TypeSafeClient({
  baseURL: 'http://127.0.0.1:8080',
  apiKey: process.env.MLX_API_KEY,
  defaultModel: 'clef-flash',
});
const result = await client.systemOne({
  state: 'The parcel is damaged.',
  questions: { damaged: { type: 'noul', instructions: 'Is the parcel damaged?' } },
});
```

Choice confidence is normalized from the most likely option; score confidence
measures expected distance from the modal level, following the
[TypeSafe formulas](https://docs.typesafe.ai/confidence). Score values are expected
zero-based indices, and `legend` maps those indices to descriptions.

`GET /v1/models` keeps the OpenAI `data` list and adds the TypeSafe `models` list.
Unknown release dates are empty strings. Decision models are excluded from chat
and agent discovery; installing CLEF does not change an existing chat default.
Aliases share model admission and execution queues. Model swaps wait for active
inference, and disconnected requests cancel their decision work.

## Validation and limits

Committed fixtures come from Cloudflare's released Python encoder and joint head;
their provenance and Apache-2.0 license are in `__test__/fixtures/clef`. Encoder
tests compare exact token IDs and spans. Head tests compare probabilities against
the PyTorch reference, allowing `1e-4` absolute error for Metal arithmetic.

The TypeScript verification scripts in `scripts/clef` exercise native inference,
repeated calls, cancellation, conversion, and the real TypeSafe SDK. Reference
fixtures were generated outside this repository using Transformers and
Cloudflare's reference implementation.
These checks validate implementation behavior, not trained-model quality.

This first implementation accepts text only, one unpadded request at a time,
with 1–256 questions, 1–255 choice options, and 2–10 score levels. The input limit
is 16,384 tokens: state is truncated to preserve the complete question schema;
a schema that cannot fit is rejected. Images and video are rejected. Prefix
caching, cross-request batching, and packed output heads are unsupported.

Both released BF16 checkpoints passed real-weight acceptance against the Python
reference, including HTTP and TypeSafe SDK requests. See the
[real-weight validation report](validation/clef-real-weights.md) for exact revisions,
probability differences, memory measurements, timings, and coverage. This does
not establish broad model quality or real-checkpoint quantization quality.
Structured numeric values use the native JSON number representation;
arbitrary-precision integers beyond 64 bits are not a supported parity case.
