import * as React from 'react';

import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';

/**
 * Chapter 1 (overview) — the autoregressive generation loop, animated.
 *
 * Self-contained: NO worker, NO model, NO WASM. The whole point of this widget
 * is to be robust as the very first thing a learner sees, while the real model
 * may still be downloading. It replays a short SCRIPTED generation so the loop
 * — forward pass → sample one token → append → repeat — is visible end to end.
 *
 * A pseudocode block highlights the line matching the current phase; the
 * candidate bars (reused TopKBars) show "the model scored every token, we kept
 * one"; the token strip grows as each sampled token is appended.
 */

const PROMPT = ['The', ' cat', ' sat', ' on', ' the'];

type Cand = { text: string; prob: number };
type Step = { cands: Cand[] };

// Each step's candidates are sorted; index 0 is the sampled (greedy) token.
const STEPS: Step[] = [
  {
    cands: [
      { text: ' floor', prob: 0.42 },
      { text: ' mat', prob: 0.19 },
      { text: ' rug', prob: 0.12 },
      { text: ' couch', prob: 0.08 },
      { text: ' bed', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: '.', prob: 0.54 },
      { text: ',', prob: 0.18 },
      { text: ' and', prob: 0.1 },
      { text: ' beside', prob: 0.05 },
      { text: ' near', prob: 0.04 },
    ],
  },
  {
    cands: [
      { text: ' It', prob: 0.31 },
      { text: ' The', prob: 0.17 },
      { text: ' A', prob: 0.09 },
      { text: ' She', prob: 0.07 },
      { text: ' Then', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: ' purred', prob: 0.27 },
      { text: ' slept', prob: 0.19 },
      { text: ' was', prob: 0.12 },
      { text: ' sat', prob: 0.08 },
      { text: ' yawned', prob: 0.06 },
    ],
  },
];

type Phase = 'forward' | 'sample' | 'append';
const PHASES: Phase[] = ['forward', 'sample', 'append'];
const TOTAL_FRAMES = STEPS.length * PHASES.length;
const FRAME_MS = 1150;

const LINES: { code: string; phase: Phase | null; comment?: string }[] = [
  { code: 'tokens = tokenize(prompt)', phase: null },
  { code: 'while (!done) {', phase: null },
  { code: '  logits = model(tokens)', phase: 'forward', comment: 'one forward pass' },
  { code: '  next   = sample(logits)', phase: 'sample', comment: 'pick one token' },
  { code: '  tokens.push(next)', phase: 'append', comment: 'append, then repeat' },
  { code: '}', phase: null },
];

export function GenerationLoop() {
  const [frame, setFrame] = React.useState(0);
  const [playing, setPlaying] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing]);

  const step = Math.floor(frame / PHASES.length);
  const phase = PHASES[frame % PHASES.length];
  const cands = STEPS[step].cands;
  const pickedIndex = 0; // index 0 is always the sampled token in the script

  // Tokens appended so far: before this step's append we have `step` tokens;
  // during/after the append phase the new one is included (and flashes).
  const appendedCount = phase === 'append' ? step + 1 : step;
  const generated = STEPS.slice(0, appendedCount).map((s) => s.cands[0].text);
  const flashIndex = phase === 'append' ? step : -1;

  const ids = cands.map((_, i) => i);
  const probs = cands.map((c) => c.prob);
  const texts = cands.map((c) => c.text);
  // Highlight the kept token only once we've "sampled" it.
  const sampledTokenId = phase === 'forward' ? -1 : pickedIndex;

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">The generation loop</div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-pressed={playing}
          className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
      </div>

      {/* Token strip — grows as tokens are appended. */}
      <div className="flex flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
        {PROMPT.map((t, i) => (
          <span
            key={`p-${i}`}
            className="rounded bg-background px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
          >
            {cleanupTokenText(t)}
          </span>
        ))}
        {generated.map((t, i) => (
          <span
            key={`g-${i}`}
            className={[
              'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
              i === flashIndex
                ? 'bg-primary/20 text-primary ring-1 ring-primary/40'
                : 'bg-primary/10 text-foreground/90',
            ].join(' ')}
          >
            {cleanupTokenText(t)}
          </span>
        ))}
        <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
          ▮
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {/* Pseudocode with the active line highlighted. */}
        <pre className="overflow-x-auto rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[11px] leading-relaxed">
          {LINES.map((ln, i) => {
            const active = ln.phase !== null && ln.phase === phase;
            return (
              <div
                key={i}
                className={[
                  'rounded px-1 transition-colors',
                  active ? 'bg-primary/15 text-foreground' : 'text-foreground/70',
                ].join(' ')}
              >
                <span>{ln.code}</span>
                {ln.comment ? <span className="text-muted-foreground"> {`// ${ln.comment}`}</span> : null}
              </div>
            );
          })}
        </pre>

        {/* Candidate distribution for this step. */}
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">
            {phase === 'forward'
              ? 'Forward pass → a score for every token'
              : phase === 'sample'
                ? 'Sample → keep one (greedy = the top bar)'
                : 'Append the kept token, then loop'}
          </div>
          <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={sampledTokenId} runKey={frame} />
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground">
        Illustrative numbers, scripted to show the loop's shape — not live output from the model.
      </p>

      <p className="text-[12px] text-foreground/85">
        An LLM is a function from a list of tokens to a score for <em>every</em> token in its vocabulary. To write more
        than one token it runs in a loop: forward pass → sample one token → append it → run again on the longer list.
        That single repeated step is all "generating text" is.
      </p>
    </div>
  );
}
