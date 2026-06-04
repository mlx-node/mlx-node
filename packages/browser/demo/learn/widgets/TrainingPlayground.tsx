import * as React from 'react';

import { Button } from '../../components/ui/button';
import {
  scatterSelfTest,
  trainInit,
  trainReset,
  trainSample,
  trainStep,
  type TinyTrainerConfig,
} from '../../lib/training-client';
import { DemoCallout } from '../inspector/DemoCallout';
import { RunButton } from '../scaffolding/RunButton';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';
import { LiveLossCurve } from './LiveLossCurve';

/**
 * Chapter 13 (Training) — LIVE training playground.
 *
 * Replaces the old scripted/illustrative demo. The learner picks a tiny text
 * corpus, presses Train, and watches a real cross-entropy loss curve drop and
 * real generated text samples improve — training a tiny char-level transformer
 * FROM SCRATCH on the actual WASM/WebGPU MLX autograd + AdamW stack. No 1.6 GB
 * model download is involved: the chapter route brings the WebGPU device up in
 * device-only mode, and this widget constructs its own ~tiny transformer via
 * the `TinyTrainer` worker hooks.
 *
 * Loop ownership lives here (per the playground spec): on Train we `trainInit`
 * once, then step in a loop that yields to the event loop between steps so the
 * UI stays responsive and Stop takes effect immediately. Every SAMPLE_EVERY
 * steps we sample a short continuation and decode it back to characters.
 */

export type TrainingPlaygroundProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

// ---------------------------------------------------------------------------
// Corpus presets. Small, highly-repetitive strings so a 2-layer/hidden-64
// transformer can visibly memorize them within a couple hundred steps.
// ---------------------------------------------------------------------------

type CorpusPreset = {
  id: string;
  label: string;
  description: string;
  text: string;
  /** Prompt prefix used for the periodic generated-text sample. */
  samplePrompt: string;
};

const CORPUS_PRESETS: CorpusPreset[] = [
  {
    id: 'pattern',
    label: 'Repeated pattern',
    description: 'A short cycle repeated many times — the easiest thing to memorize.',
    text: 'abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc',
    samplePrompt: 'abc',
  },
  {
    id: 'poem',
    label: 'Tiny rhyme',
    description: 'A nursery couplet with a small alphabet and lots of structure.',
    text: 'the cat sat on the mat. the cat sat on the mat. the cat sat on the mat.',
    samplePrompt: 'the cat ',
  },
  {
    id: 'count',
    label: 'Counting list',
    description: 'A digit sequence — the model learns to count modulo the loop.',
    text: '0123456789 0123456789 0123456789 0123456789 0123456789 0123456789',
    samplePrompt: '012',
  },
];

// ---------------------------------------------------------------------------
// Tiny model + training hyperparameters.
//
// The shape mirrors the known-good `tiny_config()` in the Rust crate's
// tiny_trainer.rs test: 2 layers, hidden 64, 4 query heads / 2 KV heads,
// headDim 16 (so numHeads*headDim == hiddenSize), intermediate 128, QK-norm on,
// tied embeddings. vocabSize is set per-corpus from the unique-char count.
// ---------------------------------------------------------------------------

const HIDDEN_SIZE = 64;
const NUM_HEADS = 4;
const NUM_KV_HEADS = 2;
const HEAD_DIM = 16; // NUM_HEADS * HEAD_DIM === HIDDEN_SIZE
const INTERMEDIATE_SIZE = 128;
const NUM_LAYERS = 2;
const MAX_POSITION_EMBEDDINGS = 64;

// Sliding training window (token ids per step). Kept comfortably under
// MAX_POSITION_EMBEDDINGS; the trainer needs >= 2 ids to have a shift target.
const WINDOW = 32;
// Sample a generated continuation every this many steps.
const SAMPLE_EVERY = 10;
// New tokens to generate per sample.
const SAMPLE_TOKENS = 24;
// Total steps a single Train press runs before auto-stopping.
const TOTAL_STEPS = 300;
const SEED = 1234;

function buildConfig(vocabSize: number): TinyTrainerConfig {
  return {
    vocabSize,
    hiddenSize: HIDDEN_SIZE,
    numLayers: NUM_LAYERS,
    numHeads: NUM_HEADS,
    numKvHeads: NUM_KV_HEADS,
    intermediateSize: INTERMEDIATE_SIZE,
    rmsNormEps: 1e-6,
    ropeTheta: 10000.0,
    maxPositionEmbeddings: MAX_POSITION_EMBEDDINGS,
    headDim: HEAD_DIM,
    useQkNorm: true,
    tieWordEmbeddings: true,
    padTokenId: 0,
    // eosTokenId -1 is an unreachable sentinel: generated ids are always >= 0,
    // so sample() never matches it and never early-stops (it runs the full
    // SAMPLE_TOKENS). bosTokenId 0 is metadata only — it's never used as an
    // index in the training/sample forward path.
    eosTokenId: -1,
    bosTokenId: 0,
  };
}

// Build a stable char<->id mapping from the corpus (unique chars, sorted for
// determinism). Returns encode/decode helpers and the vocab size.
function buildVocab(text: string) {
  const chars = Array.from(new Set(Array.from(text))).sort();
  const charToId = new Map<string, number>();
  chars.forEach((ch, i) => charToId.set(ch, i));
  const idToChar = chars;
  const encode = (s: string): number[] => {
    const out: number[] = [];
    for (const ch of Array.from(s)) {
      const id = charToId.get(ch);
      if (id !== undefined) out.push(id);
    }
    return out;
  };
  const decode = (ids: number[]): string => ids.map((id) => idToChar[id] ?? '').join('');
  return { chars, vocabSize: chars.length, encode, decode };
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

type TrainStatus =
  | { kind: 'idle' }
  | { kind: 'training' }
  | { kind: 'done' }
  | { kind: 'stopped' }
  | { kind: 'error'; error: string };

// Outcome of the one-shot WebGPU scatter_add kernel-correctness probe. The
// loss curve can't distinguish a correct accumulating gradient scatter from a
// last-write-wins bug (both still train, just differently), so we surface the
// direct invariant: a correct kernel sums three colliding +1 updates into row
// 3 (result[3] === 3.0); a broken one leaves it at 1.0.
type ScatterCheck = { kind: 'pending' } | { kind: 'pass' } | { kind: 'fail'; got: number } | { kind: 'error' };

const SCATTER_EXPECTED_ROW3 = 3;

function interpretScatterResult(result: number[]): ScatterCheck {
  // PASS requires the full shape: length 5, the colliding row summed to 3.0,
  // and every other entry ~0 (a kernel that smears writes elsewhere is also
  // wrong even if row 3 happens to read 3.0).
  const row3Ok = result.length === 5 && Math.abs((result[3] ?? NaN) - SCATTER_EXPECTED_ROW3) < 1e-3;
  const othersOk = result.length === 5 && result.every((v, i) => (i === 3 ? true : Math.abs(v) < 1e-3));
  if (row3Ok && othersOk) return { kind: 'pass' };
  return { kind: 'fail', got: result[3] ?? NaN };
}

function SliderRow({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <label htmlFor={id} className="uppercase tracking-wider">
          {label}
        </label>
        <span className="font-mono text-foreground/80">{format(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-valuetext={format(value)}
        disabled={disabled}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-muted accent-primary disabled:cursor-not-allowed disabled:opacity-50"
      />
    </div>
  );
}

// Yield to the event loop so the UI can paint and Stop clicks register between
// training steps. When the tab is visible we use requestAnimationFrame for
// smooth foreground pacing; when it's hidden (or rAF is unavailable) we fall
// back to a macrotask, since rAF is paused in a backgrounded tab — setTimeout
// keeps the loop making progress (throttled) instead of freezing.
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    const visible = typeof document === 'undefined' || document.visibilityState === 'visible';
    if (visible && typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 0);
    }
  });
}

export function TrainingPlayground({ workerRef, abortRef }: TrainingPlaygroundProps) {
  const [corpusId, setCorpusId] = React.useState(CORPUS_PRESETS[0]!.id);
  const corpus = React.useMemo(() => CORPUS_PRESETS.find((c) => c.id === corpusId) ?? CORPUS_PRESETS[0]!, [corpusId]);
  const vocab = React.useMemo(() => buildVocab(corpus.text), [corpus.text]);

  const [learningRate, setLearningRate] = React.useState(3e-3);
  const [temperature, setTemperature] = React.useState(0.5);
  const [losses, setLosses] = React.useState<number[]>([]);
  const [sampleText, setSampleText] = React.useState<string>('');
  const [running, setRunning] = React.useState(false);
  const [status, setStatus] = React.useState<TrainStatus>({ kind: 'idle' });

  // Loop control. `runGenRef` is bumped on every Train/Stop/Reset/unmount so a
  // loop iteration can detect it has been superseded and bail; `runningRef`
  // mirrors the running flag for synchronous reads inside the async loop.
  const runGenRef = React.useRef(0);
  const runningRef = React.useRef(false);
  // Per-run AbortController for the in-flight worker call, so Stop/unmount
  // rejects a pending trainStep/trainSample immediately rather than waiting
  // out the client timeout. One controller per run (not per call), bridged to
  // the app-level signal exactly once in runTraining.
  const runAbortRef = React.useRef<AbortController | null>(null);
  // Signature of the config the worker's trainer was last built with
  // (vocabSize|learningRate). When unchanged across Train presses we reuse the
  // existing trainer via trainReset instead of constructing a fresh one.
  const initedSigRef = React.useRef<string | null>(null);
  // Latest temperature, read inside the loop without re-binding it.
  const temperatureRef = React.useRef(temperature);
  React.useEffect(() => {
    temperatureRef.current = temperature;
  }, [temperature]);

  // One-shot scatter_add kernel-correctness probe. The widget only renders
  // once the device is ready (the chapter route gates it on deviceReady), so
  // the worker is available on mount. Runs exactly once: a ref latch prevents
  // a re-run if React re-mounts the effect (e.g. StrictMode double-invoke),
  // and an AbortController + cancelled flag drop the result if we unmount
  // mid-flight. Never throws into render and never blocks training.
  const [scatterCheck, setScatterCheck] = React.useState<ScatterCheck>({ kind: 'pending' });
  const scatterRanRef = React.useRef(false);
  React.useEffect(() => {
    if (scatterRanRef.current) return;
    const worker = workerRef.current;
    if (!worker) return;
    scatterRanRef.current = true;

    let cancelled = false;
    const controller = new AbortController();
    scatterSelfTest(worker, { signal: controller.signal })
      .then((result) => {
        if (cancelled) return;
        setScatterCheck(interpretScatterResult(result));
      })
      .catch((err) => {
        if (cancelled || isAbortError(err)) return;
        // Surface nothing to the learner on error (the device may not expose
        // the self-test export yet); just record it and log for diagnosis.
        setScatterCheck({ kind: 'error' });
        console.warn('[TrainingPlayground] scatter_add self-test failed', err);
      });

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [workerRef]);

  const stopLoop = React.useCallback(() => {
    runGenRef.current++;
    runningRef.current = false;
    runAbortRef.current?.abort();
    runAbortRef.current = null;
    setRunning(false);
  }, []);

  // Stop the loop on unmount so worker calls don't keep firing after the demo
  // is gone.
  React.useEffect(() => () => stopLoop(), [stopLoop]);

  async function runTraining() {
    const worker = workerRef.current;
    if (!worker) {
      setStatus({ kind: 'error', error: 'Worker is unavailable. Reload the page.' });
      return;
    }
    if (vocab.vocabSize < 2) {
      setStatus({ kind: 'error', error: 'Corpus needs at least two distinct characters.' });
      return;
    }

    // Supersede any prior loop, then claim this generation.
    stopLoop();
    const myGen = ++runGenRef.current;
    runningRef.current = true;
    setRunning(true);
    setStatus({ kind: 'training' });
    setLosses([]);
    setSampleText('');

    // One AbortController for the whole run, linked to the app-level abort
    // (worker teardown / model reload) exactly once. Every client call below
    // shares this signal, so Stop/Reset/unmount/reload cancel the in-flight
    // call without leaking a listener per step.
    const runAbort = new AbortController();
    runAbortRef.current = runAbort;
    const appSignal = abortRef.current?.signal;
    const onAppAbort = () => runAbort.abort();
    if (appSignal?.aborted) runAbort.abort();
    else appSignal?.addEventListener('abort', onAppAbort, { once: true });

    const tokens = vocab.encode(corpus.text);
    const promptIds = vocab.encode(corpus.samplePrompt);
    const config = buildConfig(vocab.vocabSize);

    // Trainer reuse: a corpus change alters the embedding shape and an LR change
    // is fixed at construction (reset(seed) does NOT change it), so either needs
    // a fresh trainer (the previous one is reclaimed by GC — <~1 MB, bounded).
    // An identical re-Train reuses the worker's trainer in place via reset(seed),
    // which re-inits weights+optimizer+step from SEED with zero new allocation.
    const sig = `${vocab.vocabSize}|${learningRate}`;

    try {
      let reused = false;
      if (initedSigRef.current === sig) {
        try {
          await trainReset(worker, SEED, { signal: runAbort.signal });
          reused = true;
        } catch (e) {
          if (isAbortError(e)) throw e; // genuine abort: bubble to the catch below
          // otherwise (e.g. "Trainer not initialized" after a worker reload):
          // fall through to trainInit and rebuild.
          initedSigRef.current = null;
        }
      }
      if (!reused) {
        await trainInit(worker, config, SEED, learningRate, { signal: runAbort.signal });
        initedSigRef.current = sig;
      }
      if (runGenRef.current !== myGen) return;

      for (let step = 0; step < TOTAL_STEPS; step++) {
        if (runGenRef.current !== myGen || !runningRef.current) break;

        // Slide a window across the corpus so every step sees a different chunk
        // (and the windows wrap, so short corpora still vary step-to-step). Use
        // span = max(1, len - WINDOW) so a corpus of length exactly WINDOW+1
        // (divisor would be 1) doesn't get stuck on a single window.
        const span = Math.max(1, tokens.length - WINDOW);
        const start = tokens.length > WINDOW ? (step * 7) % span : 0;
        const windowTokens = tokens.slice(start, start + WINDOW);
        // The trainer requires >= 2 ids (to have a shift target); WINDOW=32 and
        // every preset is far longer, and the vocabSize < 2 guard above already
        // rejected degenerate corpora, so this fallback to the full corpus is
        // belt-and-suspenders.
        const seq = windowTokens.length >= 2 ? windowTokens : tokens;

        const { loss } = await trainStep(worker, seq, { signal: runAbort.signal });
        if (runGenRef.current !== myGen) return;
        setLosses((prev) => [...prev, loss]);

        // Periodically sample a continuation so the learner watches the text
        // sharpen as the loss drops. Sampling deliberately blocks the loop and
        // is O(n·T²) (no KV cache); fine for this tiny model and kept inline for
        // simplicity and a clean generation guard.
        if ((step + 1) % SAMPLE_EVERY === 0 && promptIds.length > 0) {
          const { ids } = await trainSample(worker, promptIds, SAMPLE_TOKENS, temperatureRef.current, {
            signal: runAbort.signal,
          });
          if (runGenRef.current !== myGen) return;
          setSampleText(corpus.samplePrompt + vocab.decode(ids));
        }

        // Yield so React paints the new loss point and Stop can interrupt.
        await yieldToEventLoop();
      }

      if (runGenRef.current === myGen) {
        setStatus(runningRef.current ? { kind: 'done' } : { kind: 'stopped' });
      }
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        setStatus({ kind: 'stopped' });
      } else {
        const message = err instanceof Error ? err.message : String(err);
        setStatus({ kind: 'error', error: message });
      }
    } finally {
      // Drop the one app-signal bridge for this run.
      appSignal?.removeEventListener('abort', onAppAbort);
      if (runGenRef.current === myGen) {
        runningRef.current = false;
        setRunning(false);
        if (runAbortRef.current === runAbort) runAbortRef.current = null;
      }
    }
  }

  function handleTrainClick() {
    void runTraining();
  }

  async function handleReset() {
    stopLoop();
    setLosses([]);
    setSampleText('');
    setStatus({ kind: 'idle' });
    const worker = workerRef.current;
    if (!worker) return;
    // Reset is best-effort: it only matters if a trainer already exists. If no
    // trainer was ever created the worker replies with an error, which we
    // intentionally swallow here (the next Train press constructs one fresh).
    // Bridge this one-shot call to the app-level abort (worker teardown) and
    // tear the listener down afterwards so nothing leaks.
    const resetAbort = new AbortController();
    const appSignal = abortRef.current?.signal;
    const onAppAbort = () => resetAbort.abort();
    if (appSignal?.aborted) resetAbort.abort();
    else appSignal?.addEventListener('abort', onAppAbort, { once: true });
    try {
      await trainReset(worker, SEED, { signal: resetAbort.signal });
    } catch {
      /* no trainer yet, or torn down — nothing to reset */
    } finally {
      appSignal?.removeEventListener('abort', onAppAbort);
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Training corpus</div>
        <SegmentedToggle
          value={corpusId}
          onChange={setCorpusId}
          ariaLabel="Corpus preset"
          disabled={running}
          wrap
          options={CORPUS_PRESETS.map((preset) => ({ value: preset.id, label: preset.label }))}
        />
        <p className="text-[12px] text-foreground/80">{corpus.description}</p>
        <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-[11px] text-foreground/80">
          {corpus.text}
        </pre>
        <div className="text-[11px] text-muted-foreground">
          vocab <span className="font-mono text-foreground/80">{vocab.vocabSize}</span> chars · {NUM_LAYERS}-layer
          transformer · hidden <span className="font-mono">{HIDDEN_SIZE}</span> · trained from scratch
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <RunButton
          onClick={running ? stopLoop : handleTrainClick}
          running={false}
          label={running ? 'Stop' : 'Train'}
          disabled={vocab.vocabSize < 2}
        />
        <Button variant="outline" size="sm" onClick={() => void handleReset()} disabled={running}>
          Reset
        </Button>
        <span className="text-xs text-muted-foreground">
          Trains up to {TOTAL_STEPS} steps; samples every {SAMPLE_EVERY}.
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <SliderRow
          id="training-lr"
          label="Learning rate"
          value={learningRate}
          min={0.0005}
          max={0.01}
          step={0.0005}
          onChange={setLearningRate}
          format={(v) => v.toExponential(1)}
          disabled={running}
        />
        <SliderRow
          id="training-temperature"
          label="Sample temperature"
          value={temperature}
          min={0}
          max={1.5}
          step={0.05}
          onChange={setTemperature}
          format={(v) => v.toFixed(2)}
        />
      </div>

      {status.kind === 'error' ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Training failed.</strong> {status.error}
        </div>
      ) : null}

      <div className="space-y-2 rounded-md border border-border bg-background p-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Live training loss</div>
          <div className="text-[11px] text-muted-foreground">
            {status.kind === 'training'
              ? `training… step ${losses.length}/${TOTAL_STEPS}`
              : status.kind === 'done'
                ? 'done'
                : status.kind === 'stopped'
                  ? 'stopped'
                  : 'idle'}
          </div>
        </div>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          <strong className="font-medium text-foreground/80">This is intentional overfitting.</strong> The corpus is
          tiny and repetitive, so the model <em>memorizes</em> it within a few hundred steps and the loss falls to near
          zero. Real pretraining does the opposite — it sees each token roughly once across trillions, so it can&apos;t
          memorize and has to <em>generalize</em> (Chapter 14). The point here is to make learning <em>visible</em> in
          seconds, not to be realistic.
        </p>
        <LiveLossCurve losses={losses} maxStep={TOTAL_STEPS} />
        {scatterCheck.kind === 'fail' ? (
          <div
            role="alert"
            className="rounded-md border border-destructive/50 bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive"
          >
            ⚠ This browser computed the training gradient incorrectly, so the loss curve above may be unreliable on this
            device.{' '}
            <span className="text-destructive/80">
              (scatter_add self-test: repeated-index gradient = {scatterCheck.got}, expected 3.0)
            </span>
          </div>
        ) : null}
      </div>

      <div className="space-y-2 rounded-md border border-border bg-background p-3">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Generated sample</div>
        <div className="text-[11px] text-muted-foreground">
          Sampled continuation of <span className="font-mono">{JSON.stringify(corpus.samplePrompt)}</span>, refreshed as
          training proceeds.
        </div>
        <pre className="min-h-[2.5rem] overflow-x-auto whitespace-pre-wrap rounded-md border border-border bg-muted/40 p-2 font-mono text-[12px] text-foreground/90">
          {sampleText || (running ? 'sampling…' : 'Press Train to generate text from the tiny model.')}
        </pre>
      </div>

      <DemoCallout
        items={[
          'Watch the loss curve: from random init it starts high and drops as the model memorizes the corpus.',
          'The generated sample text gets cleaner every few steps — early samples are gibberish, later ones echo the corpus.',
          'Raise the learning rate and re-train: convergence is faster, but too high and the curve gets jumpy.',
          'This is real backprop + AdamW running on WebGPU — the same loop scales (with far more data and parameters) to frontier models.',
        ]}
      />

      <p className="text-[10px] text-muted-foreground">
        Live training in your browser — a tiny char-level transformer trained from scratch on the WASM/WebGPU MLX
        autograd + AdamW stack. No model download required.
      </p>
    </div>
  );
}
