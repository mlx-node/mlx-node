# Audio HTTP serving

`mlx serve` exposes ASR and TTS on its existing port and authentication policy.
ASR requires a converted [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
checkpoint; use the `-hf` repository, not the older `thinker_config` layout.
No separate audio package or process is required. Both audio-only and mixed
checkpoint directories are supported. Discovery reads configuration and checks
required tokenizer, generation and weight files (including complete shard sets);
it does not load weights. Missing files and unsupported VoiceDesign profiles are
excluded. Tensor contents are validated by the native loader on first use.

```sh
mlx serve --port 8000 --models-dir /path/to/models
mlx serve --port 8000 --models-dir /path/to/models --model-residency single
```

The default `capability` policy keeps one chat/decision, one ASR and one TTS
model resident, loading each lazily. Switching a TTS checkpoint preserves ASR.
`single` evicts the other capability slots before a load. Audio names match
exactly; unknown names return 404 and wrong task types return 400. Audio models
are excluded from chat model pickers and chat alias resolution.

The residency limit applies to registered model slots. Audio replacement waits
for explicit native disposal. Chat/decision replacement retains its existing
session cleanup and GC-based model lifetime, so `single` does not guarantee
that old weight allocations have been reclaimed before the next load.

Use the same `Authorization: Bearer TOKEN` or `x-api-key` as other serve APIs.
`/health` remains accessible without authentication. CORS exposes audio headers.

## Discovery

`GET /v1/models` includes audio entries with `modality: "audio"`, `task: "asr"`
or `"tts"`, and `capabilities`. `GET /v1/audio/voices?model=CHECKPOINT` returns
TTS catalogs grouped by model (omit the query to list all). Catalogs describe
`voices`, `languages`, `conditioning`, `sample_rate`, `channels`,
`text_streaming`, `audio_streaming`, `http_speech`, and `http_voice_cloning`.

Model `created` is the host catalog registration time in Unix seconds, stable
for that host's lifetime, including across lazy loads and model switches.

Sample rates come from checkpoint configuration and are checked against the
loaded model. `voice_cloning` describes the model, while `http_speech` and
`http_voice_cloning` describe this server version. Base models synthesize with a
reference voice resource. Persistent voices, VAD and WebSocket realtime sessions
are not part of this version.

## Temporary reference voices

`POST /v1/audio/voices` prepares a reference for a cloning-capable checkpoint.
It accepts multipart `audio_sample` (uncompressed WAV), `model`, `name`, and
`transcript`, all required. Optional `type` must be `audio_sample`. File and text
fields can arrive in any order. Raw `audio/wav` with those fields in the query
is also accepted. Source rates and WAV formats match the transcription endpoint;
channels are averaged to mono, and the SDK resamples for the checkpoint.

```sh
curl http://127.0.0.1:8000/v1/audio/voices \
  -F model=Base-8bit -F name=Example \
  -F audio_sample=@reference.wav -F 'transcript=The words spoken in the reference.'
```

The response includes a `Location` header and this resource:

```json
{
  "id": "voice_<uuid>",
  "object": "audio.voice",
  "type": "audio_sample",
  "name": "Example",
  "model": "Base-8bit",
  "created_at": 1791849600,
  "persistent": false,
  "status": "active"
}
```

Use the returned ID as `"voice":{"id":"voice_<uuid>"}` in speech JSON or the
NDJSON configuration. This input shape follows OpenAI's custom voice reference.
Creation is a local extension: `model` and `transcript` are required, and OpenAI
consent resources/parameters are not implemented. Unknown creation fields are
rejected. `name` is limited to 256 Unicode code points, `transcript` to 4096, and
reference audio to 30 seconds by default, in addition to the upload byte limit
and the checkpoint's own context limit. Base does not support `instructions`.
References too short for the checkpoint's speaker frontend or too long for its
encoder context return `400 invalid_audio` with `param: audio_sample`. Those
bounds are enforced by the model, not duplicated as HTTP sample-count constants.

`GET /v1/audio/voices/{id}` returns the current resource status:

- `active`: the same model instance still owns the native preparation.
- `evicted`: its native cache entry or model instance was unloaded.
- `released`: `DELETE /v1/audio/voices/{id}` explicitly released the resource.

DELETE returns `{"id":"...","object":"audio.voice.deleted","deleted":true}`;
repeating it is safe while its metadata remains available. A known inactive
voice fails synthesis with `410 voice_unavailable`; an unknown ID returns
`404 voice_not_found`. The wrong model returns `400 voice_model_mismatch`.
Aliases of the same resident instance can share a voice. Unloading its final
binding revokes the preparation; reloading the same checkpoint cannot revive it.
Queries never load weights just to inspect an inactive voice. The collection
GET remains the model capability catalog, not a list of created references.

Resources are shared by callers authorized with the serve token. There is no
separate user ownership layer. IDs are process-local and expire on restart.
Metadata is bounded (128 records by default); new creations may discard the
oldest inactive records, after which lookup returns 404. If all records are
active or reserved by creations, admission returns 429 before preparation.
Native cache capacity is separate: actual native lookup determines eviction.
Creating a voice from the same reference samples and transcript reuses the
model's prepared state instead of preparing it again. DELETE releases the ID
only; the prepared state leaves the native cache by least-recently-used
eviction.
Creation, use and deletion share the model's serial queue. Cancellation waits
for native acknowledgement and releases any late preparation result.

The service separates async resource metadata storage (`AudioVoiceStore`) from
runtime `PreparedVoice` bindings. HTTP IDs never expose native cache IDs. Only
the memory store is implemented today; reference uploads use quota-controlled
temporary files which are deleted before successful creation returns. Neither
reference audio nor prepared state is persisted. Future persistence needs both
a durable metadata/source store and a re-preparation policy for unloaded models;
the existing HTTP reference shape can remain unchanged.

## Uploaded transcription

`POST /v1/audio/transcriptions` accepts an uncompressed WAV file. Integer PCM
8/16/24/32-bit and IEEE float32/64 WAV are supported, including the corresponding
extensible formats. Channels are averaged to mono. Source sample rate is passed
to the native ASR resampler; input is not restricted to 24 kHz.

```sh
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@recording.wav -F model=ASR -F response_format=json

curl 'http://127.0.0.1:8000/v1/audio/transcriptions?model=ASR&stream=true' \
  -H 'Content-Type: audio/wav' --data-binary @recording.wav --no-buffer
```

Multipart fields can precede or follow the file. For raw `audio/wav`, use the
same field names in the query string. Supported fields:

| Field             | Meaning                                                                        |
| ----------------- | ------------------------------------------------------------------------------ |
| `model`           | Required exact checkpoint directory name                                       |
| `language`        | Omitted means automatic language detection                                     |
| `prompt`          | Optional context, at most 4096 Unicode code points                             |
| `response_format` | `json` (default) or `text`                                                     |
| `stream`          | Boolean; multipart/query use `true` or `false`                                 |
| `sampling`        | JSON object containing `max_tokens`                                            |
| `stream_options`  | JSON object containing `chunk_seconds`, `provisional_tokens`, `unfixed_chunks` |

Encode extension objects as JSON strings in multipart/query values, for example
`-F 'sampling={"max_tokens":1024}'`. Unknown ASR parameters (including timestamps,
subtitles and diarization) are rejected. Empty/malformed, compressed, misaligned,
non-finite or oversized audio is rejected. Maximum duration is 20 minutes in
addition to the byte limit; a one-shot model context budget may be tighter.

Ordinary requests use the one-shot SDK method. Without `sampling.max_tokens`, the
output budget is 10 tokens per audio second, at least 128. Audio, prompt and
output budget must fit the checkpoint context.
The server does not silently split the input or switch algorithms. JSON returns
`text`, `language`, `duration` and `finish_reason` (`stop` or `length`). Text
responses expose `X-Finish-Reason`.

The optional `language` extension in JSON and SSE uses a lowercase English
language name (for example `english`), following OpenAI's Whisper
`verbose_json` examples. Normalization happens only when serializing HTTP
responses; the native SDK retains its original language names. The field is
omitted when no language is detected. It is not a BCP-47 code or a promise of
full `verbose_json` support. Input accepts supported language codes such as
`en` as well as the SDK's language names.

`stream:true` requires `response_format:json` and uses the rolling SDK method
(default 32 output tokens per revision, configurable through `sampling`). It
starts **after upload completion**; it is not a microphone realtime session.
Each SSE event is a complete replacement snapshot:

```text
event: transcript
data: {"type":"transcript","stable":"Hello","provisional":" world","revision":2,"is_final":false,"reached_max_tokens":false,"finish_reason":null}
```

Render `stable + provisional` by replacing the previous display. Neither string
is an append-only delta, including `stable`. Revisions increase monotonically;
the final snapshot has `is_final:true`, an empty provisional string and an
explicit finish reason. A budget exhaustion is reported as `length`. A terminal
`error` event means the operation failed; no success event follows it. One-shot
and rolling results are not required to be identical.

## Text to speech

`POST /v1/audio/speech` accepts JSON:

```json
{
  "model": "CustomVoice-8bit",
  "input": "Hello. This is a local speech service.",
  "voice": "ryan",
  "response_format": "pcm",
  "speed": 1,
  "stream_format": "audio",
  "sampling": { "seed": 42, "temperature": 0.7 },
  "stream": { "chunk_duration_ms": 200, "audio_buffer_seconds": 1 }
}
```

`input` is required, nonblank and at most 4096 Unicode code points. `voice` is a
preset string, a temporary reference object `{"id":"voice_<uuid>"}`, or a VoiceDesign object such as
`{"type":"description","description":"A warm adult female voice."}`.
`instructions` is optional and checked against the checkpoint conditioning
capability; unsupported instructions produce 400. The optional extension
`language` uses the SDK's language names.

`speed` is 0.25..4, using the SDK's pitch-preserving transform. `response_format`
defaults to `wav` and only accepts `wav` or `pcm`. Other formats, including mp3,
opus, aac and flac, return `400 unsupported_response_format`. `stream_format`
defaults to `audio`; `sse` requires `response_format:pcm`.

Model controls remain in extension objects:

| Object     | Fields                                                                                  |
| ---------- | --------------------------------------------------------------------------------------- |
| `sampling` | `seed`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_duration_seconds`   |
| `stream`   | `chunk_duration_ms`, `audio_buffer_seconds`, `max_segment_graphemes`, `queued_segments` |

`response_format: wav` is sent after synthesis finishes, so its `chunk_duration_ms`
defaults to 640 ms (the SDK's non-streaming default); `pcm` keeps 160 ms. Each
segment's first chunk is one 80 ms codec frame, or two when the bootstrap frame
may be dropped.

`pcm` and SSE output drop the model's silent bootstrap frame (80 ms) from each
segment, as sglang-omni does for streamed requests. Only voice `ryan` with
`language` `english`, no `instructions` and no `sampling` field other than `seed`
and `max_duration_seconds` is eligible. The first frame is dropped when its RMS
is at most 0.001 and its peak at most 0.0032; a NaN sample keeps it. `wav`
always keeps the frame. The top-level boolean `suppress_bootstrap_silence`
(also in the NDJSON config record) set to `false` keeps it in `pcm`; `true` is
the default and does not force the drop; another type is a 400. `pcm` here is
always streamed, while sglang-omni keeps the frame for `pcm` without
`stream:true`. sglang-omni also keeps it above 24 live streams, for
`initial_codec_chunk_frames` 0 or 16 and up, and for `stream_codec_output:false`;
this server runs one synthesis per model and has neither field.

A JSON `input` is synthesized as one sequence, so `max_duration_seconds`
(0.1..163.84, default 163.84 = 2,048 codec frames) limits the whole request. On
the NDJSON text-stream route it limits **each committed segment** (one per
sentence), and `max_segment_graphemes` and `queued_segments` apply only there.
Truncation is reported as `finish_reason:length` (normal TTS completion is `eos`).
Defaults otherwise follow the SDK. Unknown nested controls are rejected.

ASR prompts and TTS text, instructions, reference transcripts and voice descriptions are encoded as
literal content. Checkpoint control-token spellings inside these fields cannot
introduce template boundaries. This does not prevent ordinary language from
influencing model output; prompts and instructions are intended to do that.

### Output and statistics

PCM is signed 16-bit little-endian, interleaved by channel. Read its actual
sample rate and channels; unlike OpenAI PCM, it is not fixed to 24 kHz.

| Output | Initial headers                                                        | Completion                                                              |
| ------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| WAV    | All five statistics below, `X-Finish-Reason`, valid `Content-Length`   | Legal finalized RIFF/WAV file                                           |
| PCM    | `X-Sample-Rate`, `X-Channels`, `X-First-Pcm-Ms`, `Trailer` declaration | `X-Audio-Seconds`, `X-Realtime-Factor`, `X-Finish-Reason` HTTP trailers |
| SSE    | Sample rate, channels, first PCM time                                  | Statistics in `speech.audio.done`                                       |

WAV synthesis writes to a quota-controlled temporary file and finalizes its
header before sending a response. Its download does not hold the model execution
slot. PCM and SSE commit headers once the first output chunk is ready, preserving
correct HTTP error statuses for early failures.

After PCM headers are sent, a failure aborts the HTTP response. The status
cannot change and there is no JSON error frame or successful completion trailer
in that partial audio. Treat a response-body read error or incomplete HTTP
message as failure, discard the partial result, and do not automatically replay
speech that may already have played. Absence of trailers alone is not a reliable
failure signal because proxies can remove them. Use SSE when a structured late
error is required. Default SDK retries during request establishment are distinct
from retries implemented by an application's audio-reading loop.

SSE sends `speech.audio.delta` with `audio` (base64 PCM), `response_format:pcm`,
`sample_rate`, and `channels`. `speech.audio.done` carries `audio_seconds`,
`first_pcm_ms`, `realtime_factor`, `finish_reason`, `input_wait_ms`, and
`consumer_wait_ms`. Unlike ASR revisions, these audio chunks are concatenated.

Audio seconds measures actual **post-speed** output frames divided by sample
rate. First PCM time includes upload, input wait, queue wait and model loading.
RTF measures native synthesis plus speed processing divided by output duration;
network and input waits are excluded. An RTF below one is faster than realtime.
Browser clients needing final statistics should use SSE; browser Fetch does not
expose HTTP trailers reliably. PCM trailers can also be stripped by proxies.
The tested official Node SDK does not expose these trailers through its
streaming API, so raw PCM cannot deliver final duration, RTF or finish reason
through that client. Choose SSE or WAV HTTP response headers when those
values are required. Trailers are an optional facility for capable HTTP clients.

## Incremental text input (Node/backend)

`POST /v1/audio/speech/stream` accepts `application/x-ndjson`. The first nonblank
record must be a config object with the speech options except `input`:

```ndjson
{"type":"config","model":"CustomVoice-8bit","voice":"ryan","response_format":"pcm"}
{"type":"text","text":"Hello. "}
{"type":"flush"}
"More text can arrive later."
{"type":"instruct","value":null}
```

Subsequent records are strings or SDK `text`, `flush`, `instruct` events. An
instruct value is a string or null, checked against model capabilities. A normal
request EOF flushes the remaining text. All text events together have a 4096
code point limit. Config is only allowed first; each record is validated as read.

The response is PCM16 only. Input and output must be consumed concurrently; see
[the Node example](../examples/tts/http-stream.ts). This version does not promise
browser Fetch full-duplex compatibility. A malformed late record, cancellation
or inference failure terminates the PCM transport, without inserting JSON into
audio or sending successful completion trailers. A client should treat an
incomplete HTTP response as failure and must not automatically replay already
played speech.

## Bounds, scheduling and shutdown

Each model object has one serial execution queue; aliases share the queue.
The normal serve queue limit is 16 waiters. Model loads/disposals take the host's
exclusive coordinator lock; inference holds shared ownership. Queued requests
hold no inference lock, and recheck their binding after entering. Cold load
admission transfers to the resident queue without double counting. A queued
request whose model was replaced receives `429 model_changed`; clients can retry
before any output has been emitted.

Official SDKs retry eligible 429 responses by default. The service's capacity
and queue rejections happen before that request begins inference: a rejected
attempt produces no audio, and a later accepted retry runs the synthesis once.
The fixed-input SDK tests cover this with both clients' default retry policies.
Once work is handed to the ASR/TTS SDK, errors before HTTP headers are committed
include `X-Should-Retry: false` and omit `Retry-After`, retaining their actual
HTTP status. This conservative boundary also covers synchronous SDK start
failures, failures before the first output and WAV storage failures after
synthesis begins. Validation, admission and model loading happen before this
boundary and retain their existing retry behavior.

`X-Should-Retry` is a nonstandard compatibility hint honored by the official
OpenAI SDKs (tested with the Node SDK), not an HTTP-wide guarantee. The service does not
deduplicate idempotency keys; connection failures before receiving the error
headers, clients ignoring the hint or application retry loops can repeat work.
For calls that must avoid SDK resubmission, set `maxRetries: 0` (Node) or
`max_retries=0` (Python). Do not add an automatic replay loop around partial audio.
Errors after streaming starts still use an SSE terminal error or a broken PCM
transport; headers can no longer change. CORS exposes both retry headers.

Additional settings are available through `createInferenceHost({ audio: ... })`
or `createServer({ audio: ... })`:

| Setting                 | Default                                                |
| ----------------------- | ------------------------------------------------------ |
| `maxUploadBytes`        | 25 MiB WAV                                             |
| `maxTempBytes`          | 512 MiB across uploads and WAV downloads               |
| `maxConcurrentRequests` | 17 active audio requests, including uploads            |
| `inputIdleMs`           | 30000 ms while actually waiting for input              |
| `outputDrainMs`         | 30000 ms of continuous output backpressure             |
| `maxVoiceRecords`       | 128 voice metadata records, including inactive records |
| `maxReferenceSeconds`   | 30 seconds per reference upload                        |

The concurrency cap covers ASR, TTS and voice resource operations together. The temporary storage quota is
an independent hard bound: it covers retained ASR/reference uploads and TTS WAV spools,
including completed WAVs still being downloaded. Exhaustion returns
`503 audio_storage_full` before response headers; late failures follow the
transport rules above. Size it for the expected mix: maximum retained upload
bytes plus expected concurrent WAV output bytes (`seconds * sample_rate *
channels * 2`, plus headers), with headroom. For example, 17 maximum-size uploads
use 425 MiB, leaving 87 MiB in the default quota. This is a sizing example, not a
constructor constraint: `concurrency * maxUploadBytes` does not bound variable
TTS output sizes, and intentionally smaller quotas are valid.

NDJSON is capped at 64 KiB per record and 1 MiB per request. Multipart uses a
streaming parser with one file, 16 fields and bounded field sizes. Capacity
rejection is 429 plus `Retry-After`; uploads over the byte limit are 413.
Temporary storage exhaustion is 503: upload failures before SDK dispatch include
`Retry-After`; WAV output failures after dispatch include `X-Should-Retry: false`.

PCM/SSE writes wait for `drain`, stopping generator consumption. Native PCM and
text segmentation remain bounded by SDK options. Time spent intentionally
pausing reads for backpressure is excluded from input idle time. Logs contain
the route pathname, byte counts and response metadata, not the original query,
WAV, PCM, base64 or full input text.

Disconnect cancellation is cooperative at native safe points. Inference locks
and queue ownership remain held until native exit is acknowledged; an in-flight
Metal operation cannot be instantly preempted. Shutdown stops admission,
cancels tasks, waits for native completion, then releases audio models and
removes temporary files. Normal upload completion is not a disconnect.
On server shutdown, an uncommitted request receives `503 server_shutdown`, SSE
receives one terminal error event, and an active binary stream is terminated.
Keep-alive transports close after terminal responses have flushed.

## Reverse proxy

Disable both request and response buffering for full-duplex NDJSON:

```nginx
location /v1/audio/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_request_buffering off;
    proxy_buffering off;
    proxy_read_timeout 300s;
}
```

SSE responses also set `X-Accel-Buffering: no`. Ensure any upstream gateway does
not compress or buffer events. Use SSE if a proxy cannot preserve final trailers.

## Validation

Normal tests use substitute inference backends and the native audio codecs;
no checkpoint download is needed. The server test suite under
`__test__/server/audio-*.test.ts` exercises the HTTP/SSE/NDJSON protocol and
official OpenAI Node SDK compatibility (including multipart uploads, binary
audio, SSE events, error objects, and duplex NDJSON).

The protocol intentionally follows OpenAI field names while exposing its
revision semantics and checkpoint-dependent PCM rate. References:
[OpenAI speech API](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create),
[Node HTTP trailers](https://nodejs.org/api/http.html#responseaddtrailersheaders),
[Busboy streaming multipart](https://github.com/fastify/busboy).
