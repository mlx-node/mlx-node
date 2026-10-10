# CLEF decisions

CLEF and CLEF Flash are text decision models built on the dense Qwen3.5/Qwen3.8
backbone. They answer `noul` (yes/no probability), `choice`, and `score` questions
in one backbone pass. They are exposed separately from chat models.

## Checkpoints

A checkpoint directory must contain the Qwen config, tokenizer, backbone weights,
`joint_head_config.json`, and `joint_head.safetensors`. Model discovery identifies
CLEF from both head assets. An incomplete head is an error.

The regular model converter copies both head assets unchanged. Backbone
quantization is supported by the existing Qwen loader; the output `lm_head` must
remain dense, and tied output embeddings are unsupported. A custom quantization
recipe that packs `lm_head` will be rejected when loading CLEF.

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
