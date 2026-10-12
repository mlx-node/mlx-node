import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';

import { loadTtsModel, readWav, WavWriter, createAudioPlayer, changeAudioSpeed } from '@mlx-node/tts';
import type { TtsInput, AudioPlayer, TtsVoice } from '@mlx-node/tts';

import { decodeTtsInput, ttsFileBytes } from './tts-input.js';

export async function run(args: string[]): Promise<void> {
  const { values } = parseArgs({
    args,
    options: {
      help: { type: 'boolean', short: 'h' },
      model: { type: 'string', short: 'm' },
      text: { type: 'string', short: 't' },
      file: { type: 'string', short: 'i' },
      output: { type: 'string', short: 'o' },
      play: { type: 'boolean' },
      voice: { type: 'string' },
      instruct: { type: 'string' },
      'instruct-file': { type: 'string' },
      'voice-description': { type: 'string' },
      'input-format': { type: 'string', default: 'text' },
      language: { type: 'string' },
      'reference-audio': { type: 'string' },
      'reference-text': { type: 'string' },
      'reference-text-file': { type: 'string' },
      'chunk-ms': { type: 'string' },
      'prebuffer-seconds': { type: 'string' },
      'buffer-seconds': { type: 'string' },
      'max-duration': { type: 'string' },
      seed: { type: 'string' },
      speed: { type: 'string', default: '1' },
    },
  });
  if (values.help) {
    console.log(`Usage: mlx tts -m <model path> [--text <text> | --file <file> | stdin]
  --voice <id>                CustomVoice preset (see model config)
  --reference-audio <wav>     Base voice-cloning reference
  --reference-text <text>     Exact transcript of the reference audio
  --reference-text-file <txt> Read the reference transcript from a file
  --voice-description <text>  VoiceDesign voice description (1.7B)
  --instruct <text>           Delivery instruction (supported voice modes only)
  --instruct-file <file>      Read the delivery instruction from a UTF-8 file
  --input-format text|jsonl   Incremental input format (default: text; 64 KiB per JSONL record;
                              with jsonl, --text is parsed as a single JSONL record)
  --language <language>      Default: auto
  --play                     Stream PCM to the default macOS output device
  -o, --output <wav>          Stream to a PCM16 WAV file
  --chunk-ms <ms>             Target PCM chunk duration (default: 160)
  --prebuffer-seconds <s>     Playback prebuffer (default: 0.32; at most buffer capacity)
  --buffer-seconds <s>        Playback buffer capacity (default: 1)
  --max-duration <seconds>    Per-segment generation limit (default: 120); truncation is reported
  --seed <integer>            Reproducible sampling
  --speed <factor>            Pitch-preserving speech speed (0.25..4; default: 1)

Text from stdin is decoded incrementally and submitted at sentence/phrase boundaries.
At least one of --play or --output is required. Playback and saving may be combined.
TTS_MLX_CACHE_LIMIT=<GiB> limits the shared MLX free-buffer pool while TTS is loaded;
fractional values are supported, and unset/0 uses the normal cache policy.`);
    return;
  }
  if (!values.model || (!values.play && !values.output))
    throw new Error('--model and at least one of --play or --output are required');
  if (values.text !== undefined && values.file !== undefined) throw new Error('Choose --text or --file');
  if (values.text === undefined && values.file === undefined && process.stdin.isTTY)
    throw new Error('Supply --text, --file, or piped stdin');
  if (values['reference-text'] && values['reference-text-file'])
    throw new Error('Choose one reference transcript source');
  if (
    [values.voice, values['reference-audio'], values['voice-description']].filter((v) => v !== undefined).length !== 1
  )
    throw new Error('Choose exactly one of --voice, --reference-audio, or --voice-description');
  if (values.instruct !== undefined && values['instruct-file'] !== undefined)
    throw new Error('Choose --instruct or --instruct-file');
  const format = values['input-format'];
  if (format !== 'text' && format !== 'jsonl') throw new Error('--input-format must be text or jsonl');
  const instruct =
    values['instruct-file'] === undefined ? values.instruct : await readFile(values['instruct-file'], 'utf8');
  const referenceText = values['reference-text-file']
    ? await readFile(values['reference-text-file'], 'utf8')
    : values['reference-text'];
  if (Boolean(values['reference-audio']) !== Boolean(referenceText))
    throw new Error('Voice cloning requires both reference audio and its transcript');
  const controller = new AbortController();
  let player: AudioPlayer | undefined;
  let writer: WavWriter | undefined;
  const cancel = () => {
    controller.abort();
    player?.cancel();
  };
  const started = performance.now();
  const model = await loadTtsModel(values.model);
  process.once('SIGINT', cancel);
  console.error(
    `Loaded ${model.capabilities.family}/${model.capabilities.variant} in ${(performance.now() - started).toFixed(0)} ms`,
  );
  try {
    let voice: TtsVoice;
    if (values['reference-audio'] && referenceText) {
      const preparing = performance.now();
      voice = await model.prepareVoice({ audio: await readWav(values['reference-audio']), transcript: referenceText });
      console.error(`Prepared voice in ${(performance.now() - preparing).toFixed(0)} ms`);
    } else if (values['voice-description'] !== undefined) {
      voice = { type: 'description', description: values['voice-description'] };
    } else {
      if (!values.voice)
        throw new Error(`Supply --voice (${model.capabilities.voices.join(', ')}) or a reference for a Base model`);
      voice = values.voice;
    }
    const input: TtsInput =
      values.text !== undefined && format === 'text'
        ? values.text
        : decodeTtsInput(
            values.text !== undefined
              ? (async function* () {
                  yield Buffer.from(values.text!);
                })()
              : values.file
                ? ttsFileBytes(values.file)
                : process.stdin,
            format,
          );
    if (values.output)
      writer = await WavWriter.open(values.output, model.capabilities.sampleRate, model.capabilities.channels);
    if (values.play)
      player = await createAudioPlayer(model.capabilities.sampleRate, model.capabilities.channels, {
        prebufferSeconds: values['prebuffer-seconds'] === undefined ? undefined : Number(values['prebuffer-seconds']),
        bufferSeconds: values['buffer-seconds'] === undefined ? undefined : Number(values['buffer-seconds']),
      });
    const stream = model.synthesizeStream(input, {
      voice,
      instruct,
      language: values.language,
      signal: controller.signal,
      chunkDurationMs: values['chunk-ms'] === undefined ? undefined : Number(values['chunk-ms']),
      maxDurationSeconds: values['max-duration'] === undefined ? undefined : Number(values['max-duration']),
      seed: values.seed === undefined ? undefined : Number(values.seed),
    });
    const processed = changeAudioSpeed(stream, Number(values.speed));
    for await (const chunk of processed) {
      await writer?.write(chunk);
      await player?.write(chunk.samples);
    }
    const synthesis = await stream.completed;
    const outputSeconds = processed.stats.outputFrames / model.capabilities.sampleRate;
    console.error(
      JSON.stringify(
        {
          synthesis,
          audioProcessing: {
            ...processed.stats,
            speed: Number(values.speed),
            outputSeconds,
            realTimeFactor: outputSeconds
              ? (synthesis.synthesisMs + processed.stats.processingMs) / 1000 / outputSeconds
              : null,
          },
          playback: await player?.finish(),
        },
        null,
        2,
      ),
    );
  } catch (error) {
    if (!controller.signal.aborted) throw error;
    // Ctrl+C landed: the stream rejects with AbortError, but that is a clean
    // cancellation, not a failure — report the conventional 130 and let the
    // finally block still tear down player, writer, and model.
    process.exitCode = 130;
    console.error('Cancelled');
  } finally {
    player?.cancel();
    try {
      await writer?.close();
    } finally {
      await model.dispose();
      process.removeListener('SIGINT', cancel);
    }
  }
}
