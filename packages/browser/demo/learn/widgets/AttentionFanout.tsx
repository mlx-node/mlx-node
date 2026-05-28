import * as React from 'react';

import type { AttentionLayer, AttentionRun } from '../../../src/inspector-types';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../components/ui/select';

/**
 * Chapter 3 fan-out diagram — reinterpret one row of the attention matrix
 * as a graph: the last query token in the centre, weighted edges fanning
 * back to each key token. Line thickness + opacity encode the attention
 * weight (post-softmax).
 *
 * Why this widget exists: the heatmap shows the whole matrix; this widget
 * isolates the moment where matrix → graph clicks. The last row of the
 * heatmap is the one that decides the next token, so showing only that
 * row as messages-back-to-the-context makes attention's "weighted message
 * passing" interpretation visceral.
 *
 * Inspired by the encoder-s3 fan-out panel in An-Jhon/Hand-Drawn-Transformer.
 * Real model data — uses the same `AttentionRun.attention[layer].scores`
 * the heatmap consumes.
 */

function renderToken(text: string): string {
  if (text.startsWith(' ')) return '·' + text.slice(1);
  if (text === '\n') return '↵';
  if (text === '\t') return '⇥';
  return text;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function AttentionFanout({ run }: { run: AttentionRun }) {
  const fullLayers = run.attention.filter((l) => l.kind === 'full' && l.scores.length > 0);
  const [layerIdx, setLayerIdx] = React.useState<number>(
    fullLayers.length > 0 ? fullLayers[fullLayers.length - 1]!.layerIndex : 0,
  );
  const [headIdx, setHeadIdx] = React.useState<number>(0);

  // Re-clamp when the run changes (different prompt → possibly fewer layers).
  React.useEffect(() => {
    if (fullLayers.length === 0) return;
    if (!fullLayers.some((l) => l.layerIndex === layerIdx)) {
      setLayerIdx(fullLayers[fullLayers.length - 1]!.layerIndex);
    }
  }, [run, fullLayers, layerIdx]);

  const layer: AttentionLayer | undefined = fullLayers.find((l) => l.layerIndex === layerIdx);
  React.useEffect(() => {
    if (layer && headIdx >= layer.numHeads) setHeadIdx(0);
  }, [layer, headIdx]);

  if (!layer) {
    return (
      <div className="rounded-md border border-dashed border-border bg-background p-4 text-xs text-muted-foreground">
        No full-attention layers captured for this run.
      </div>
    );
  }

  const seqLen = run.tokens.length;
  const lastRow = seqLen - 1;
  const stride = seqLen * seqLen;
  const headOffset = headIdx * stride + lastRow * seqLen;
  const weights: number[] = [];
  for (let k = 0; k < seqLen; k++) {
    weights.push(layer.scores[headOffset + k] ?? 0);
  }
  const maxW = Math.max(...weights, 1e-9);

  // Layout: tokens along a top row, target query token at the bottom-center,
  // edges fan from each top node down to the target. Widen as the prompt grows
  // so labels don't crash into each other.
  const PAD_X = 18;
  const TOP_Y = 30;
  const BOT_Y = 175;
  const TOKEN_W = Math.max(64, Math.min(110, 520 / seqLen));
  const W = Math.max(360, PAD_X * 2 + TOKEN_W * seqLen);
  const H = 220;

  function tokenX(i: number): number {
    if (seqLen === 1) return W / 2;
    const span = W - PAD_X * 2 - TOKEN_W;
    return PAD_X + TOKEN_W / 2 + (i / (seqLen - 1)) * span;
  }

  const queryTokenText = run.tokens[lastRow]?.text ?? '?';
  const targetX = W / 2;

  // Highlight the strongest edge so the eye finds it without scanning.
  let argmax = 0;
  for (let i = 1; i < weights.length; i++) {
    if ((weights[i] ?? 0) > (weights[argmax] ?? 0)) argmax = i;
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Attention as messages · last token's view
        </div>
        <div className="flex items-center gap-2">
          <Select value={`${layerIdx}`} onValueChange={(v) => setLayerIdx(Number(v))}>
            <SelectTrigger size="sm" className="min-w-[7rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {fullLayers.map((l) => (
                <SelectItem key={l.layerIndex} value={`${l.layerIndex}`}>
                  Layer {l.layerIndex}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={`${headIdx}`} onValueChange={(v) => setHeadIdx(Number(v))}>
            <SelectTrigger size="sm" className="min-w-[6rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Array.from({ length: layer.numHeads }, (_, i) => (
                <SelectItem key={i} value={`${i}`}>
                  Head {i}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The last query token <span className="font-mono">"{renderToken(queryTokenText)}"</span> is the one that decides
        the next word. Each line is the attention weight from that token back to one earlier token — thicker means more
        weight. The brightest line is what the model is "listening to" most.
      </p>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="block h-auto w-full"
        role="img"
        aria-label="Fan-out diagram showing attention weights from the last token to all earlier tokens"
      >
        {/* edges */}
        {weights.map((w, i) => {
          const x = tokenX(i);
          const t = w / maxW;
          const isPeak = i === argmax;
          const strokeWidth = lerp(0.6, 4.2, t);
          const opacity = lerp(0.18, 0.95, t);
          return (
            <g key={i}>
              <line
                x1={x}
                y1={TOP_Y + 10}
                x2={targetX}
                y2={BOT_Y - 12}
                stroke={isPeak ? 'oklch(0.7 0.18 25)' : 'currentColor'}
                strokeOpacity={opacity}
                strokeWidth={strokeWidth}
                strokeLinecap="round"
              />
              {/* weight label, mid-edge, only when the weight is non-trivial */}
              {t > 0.04 ? (
                <text
                  x={lerp(x, targetX, 0.65)}
                  y={lerp(TOP_Y + 10, BOT_Y - 12, 0.65) - 2}
                  fontSize={9}
                  fill="currentColor"
                  fillOpacity={Math.min(1, opacity + 0.1)}
                  textAnchor="middle"
                >
                  {w.toFixed(2)}
                </text>
              ) : null}
            </g>
          );
        })}

        {/* top-row token chips */}
        {run.tokens.map((tok, i) => {
          const x = tokenX(i);
          const isPeak = i === argmax;
          return (
            <g key={i}>
              <rect
                x={x - TOKEN_W / 2 + 4}
                y={TOP_Y - 12}
                width={TOKEN_W - 8}
                height={22}
                rx={5}
                fill={isPeak ? 'oklch(0.7 0.18 25 / 0.15)' : 'currentColor'}
                fillOpacity={isPeak ? 1 : 0.08}
                stroke={isPeak ? 'oklch(0.7 0.18 25)' : 'currentColor'}
                strokeOpacity={isPeak ? 0.7 : 0.25}
              />
              <text x={x} y={TOP_Y + 3} fontSize={11} textAnchor="middle" fill="currentColor" fillOpacity={0.9}>
                {renderToken(tok.text)}
              </text>
            </g>
          );
        })}

        {/* bottom target chip */}
        <g>
          <rect
            x={targetX - TOKEN_W / 2 - 8}
            y={BOT_Y - 12}
            width={TOKEN_W + 16}
            height={26}
            rx={6}
            className="fill-primary/15"
            stroke="oklch(0.7 0.18 25)"
            strokeOpacity={0.55}
          />
          <text x={targetX} y={BOT_Y + 5} fontSize={12} textAnchor="middle" className="fill-primary" fontWeight={600}>
            {renderToken(queryTokenText)}
          </text>
          <text
            x={targetX}
            y={BOT_Y + 26}
            fontSize={9}
            textAnchor="middle"
            fill="currentColor"
            fillOpacity={0.55}
          >
            query · position {lastRow}
          </text>
        </g>
      </svg>

      <div className="text-[11px] text-muted-foreground">
        Strongest pull: <span className="font-mono text-foreground/85">{renderToken(run.tokens[argmax]?.text ?? '?')}</span> at position{' '}
        {argmax} with weight <span className="font-mono text-foreground/85">{(weights[argmax] ?? 0).toFixed(3)}</span>. Weights along the row sum
        to <span className="font-mono">{weights.reduce((a, b) => a + b, 0).toFixed(2)}</span>.
      </div>
    </div>
  );
}
