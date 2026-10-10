# CLEF support

## Scope and completion gates

Implement text-only CLEF and CLEF Flash using the existing Qwen3.5 backbone,
a native joint schema head, an asynchronous public `ClefModel`, and the
Jev-compatible `POST /v1/systemone` endpoint. Reject media explicitly.

1. Research: compare Cloudflare's released `joint_schema_model.py` and head
   configuration against the current Qwen loader, model thread, conversion,
   and server lifecycle. GitHub source investigation is delegated through mlx.
2. Encoding and checkpoints: preserve fragment boundaries, question insertion
   order, sorted option names and semantic JSON; retain both head assets.
3. Native inference: execute the backbone and head on the same model thread,
   use request-local state, retain every final hidden vector, validate head
   tensors, and gather untied output embeddings. Initially require dense output
   embeddings; reject packed output heads rather than interpret packed storage.
4. Serving: add decision registration without chat sessions, bounded admission,
   shared process work coordination, disconnect cancellation, model listing,
   request IDs and TypeSafe confidence formulas.
5. Verification: reference encoder fixtures, independently generated tiny head
   fixtures, malformed checkpoint/request coverage, lifecycle/alias tests,
   native build, TypeScript checks, relevant tests and repository checks.
6. Review: inspect numerical operations, queue cleanup, cancellation and model
   replacement; fix findings and rerun affected checks.

## Sources

- Shared research: https://chatgpt.com/share/6aca049b-1b2c-83ec-b993-dd8fc0194f27
- Reference: https://huggingface.co/Cloudflare/clef-flash/blob/main/joint_schema_model.py
- HTTP contract: https://docs.typesafe.ai/api
- Confidence: https://docs.typesafe.ai/confidence

## Known limitations

Images, video, cross-request batching and prefix caching are follow-up work.
Both released BF16 checkpoints have now passed the acceptance cases below.
Broader accuracy evaluation and real-checkpoint quantization quality remain
follow-up work; these functional cases are not a model-quality benchmark.

## Implementation and review evidence

- Added the native encoder and joint head, request-local Qwen backbone execution,
  typed asynchronous API, conversion asset preservation, decision discovery and
  registration, and `/v1/systemone` serving.
- Compared encoder tokens/spans against the released Python implementation and
  head probabilities against independently generated PyTorch fixtures. The head
  error was `2.98e-8` on CPU and `4.45e-5` on Metal; the committed test uses Metal
  with a `1e-4` tolerance.
- A four-layer hybrid Qwen fixture (three recurrent layers and one full-attention
  layer) plus the full joint-head topology matched Python within `0.000606`
  maximum probability error over a 688-token input. The integration script also
  passed repeated requests, cancellation, lossless conversion, an 8-bit backbone,
  incomplete-checkpoint rejection, native host discovery/loading, and TypeSafe SDK
  0.6.0 HTTP calls and model listing.
- Reviewed numerical operations, schema ordering, alias admission, cancellation,
  and model replacement. Fixed admission transfer during cold loading, malformed
  head asset detection, advertised context limits, and disabling chat cache policy
  before checkpoint loading begins.
- Native release packaging, TypeScript compilation, targeted lint, four native
  CLEF tests, and 237 targeted server/model tests passed. Fixture-dependent tests
  were rerun after restoring existing local tokenizer and dataset assets: 229
  passed, with the separate agent CLI smoke still failing on its removed
  `--no-session` argument. The broad run also exposed a dashboard npm-pack output
  failure. Repository formatting is blocked by eight untouched files.
- The final broad run, excluding those two identified failures, passed 3,997
  tests. Three additional suites initially lacked the complete Qwen test
  checkpoint; after cloning the existing local fixture, all 18 tests in those
  suites passed. Twenty optional test files remained skipped. The final native
  rebuild and end-to-end script also passed with the global persistence flag
  enabled, exercising CLEF's explicit cache-policy override.
- The requested GitHub investigation was dispatched through `mlx delegate` for
  PR #196, but returned no findings before it was interrupted. Implementation
  research used local source and the released Hugging Face reference.

Both checkpoints were subsequently downloaded with `yarn mlx` and tested against
the released Python reference. All 13 decisions and input token counts matched
for each model; maximum probability errors were 0.000786 (Flash) and 0.002310
(CLEF). Native HTTP/SDK, cancellation recovery and repeat checks passed. See the
[real-weight evidence](../validation/clef-real-weights.md). Python reference scripts
were moved outside the repository at the user's request.

See [CLEF usage and limits](../clef.md) for the public surface and remaining work.
