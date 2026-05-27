import * as React from "react";

import { Button } from "../../components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";
import { Textarea } from "../../components/ui/textarea";
import type { AttentionLayer, AttentionRun } from "../../../src/inspector-types";
import { runForInspector } from "../../lib/inspector-client";
import {
  AttentionHeatmap,
  drawAttentionHeatmap,
} from "../inspector/AttentionHeatmap";
import { Prose } from "../Prose";
import { makeMockAttentionRun } from "../mock-data";

const DEFAULT_PROMPT = "The cat sat on the mat.";
const QWEN_NUM_HEADS = 8;
const QWEN_NUM_KV_HEADS = 2;
const QWEN_HEAD_DIM = 256;
// Realistic decode context length; makes the KV-cache savings tangible (~MB scale).
const ASSUMED_DECODE_SEQ_LEN = 4096;

const GROUP_COLORS = [
  { fill: "#0ea5e9", text: "#f0f9ff" },
  { fill: "#f43f5e", text: "#fff1f2" },
  { fill: "#10b981", text: "#ecfdf5" },
  { fill: "#f59e0b", text: "#fffbeb" },
  { fill: "#8b5cf6", text: "#f5f3ff" },
  { fill: "#06b6d4", text: "#ecfeff" },
  { fill: "#ec4899", text: "#fdf2f8" },
  { fill: "#84cc16", text: "#f7fee7" },
] as const;

export function MultiheadGqaChapterBody() {
  return (
    <Prose>
      <h1>Multi-head attention &amp; GQA</h1>
      <p>
        Chapter 3 introduced attention as a single operation:{" "}
        <code>softmax(QKᵀ/√d)·V</code>. In real LLMs, that operation is run{" "}
        <em>many times in parallel</em>, once per <strong>head</strong>, and the
        results are concatenated and projected back together. The reason is
        capacity: a single head can model one type of relationship at a time —
        say, "look at the previous token." Multiple heads in parallel let the
        model attend to different patterns simultaneously: one head can chase
        the previous token, another the matching bracket, another the
        subject—verb agreement at distance.
      </p>

      <h2>Standard multi-head attention (MHA)</h2>
      <p>
        For hidden size <code>d</code> and <code>H</code> heads, each head has
        its own slice of dimension <code>d_head = d / H</code>. The model
        projects the token vector to a Q, K and V of shape{" "}
        <code>[H, seq, d_head]</code> by using three full <code>d × d</code>{" "}
        weight matrices, then reshaping. Each head runs its own attention over
        its own Q/K/V slice; the <code>H</code> outputs are concatenated back
        into a <code>d</code>-wide vector that flows on through the layer.
      </p>
      <p>
        At inference, K and V for each token are <strong>cached</strong> across
        generation steps — that's what makes the second token onward fast. The
        cache size is <code>2 × H × d_head × seq_len</code> floats{" "}
        <em>per layer</em>. For a 32k-context model with deep layer counts and
        wide hidden sizes, that's gigabytes — and unlike weights, it grows with
        each new token. <strong>The KV cache, not the weights, is what
        dominates memory at long context.</strong>
      </p>

      <h2>Grouped query attention (GQA)</h2>
      <p>
        GQA is the now-standard trick to shrink the cache. Instead of giving
        every query head its own K/V head, you partition the query heads into{" "}
        <code>num_kv_heads</code> groups and let every head in a group share a
        single K/V pair:
      </p>
      <pre>
        <code>{`num_heads     = H            (e.g. 8 for Qwen3.5-0.8B)
num_kv_heads  = G            (e.g. 2 for Qwen3.5-0.8B)
group_size    = H / G        (4 query heads per K/V head)`}</code>
      </pre>
      <p>
        The Q projection still has <code>H</code> heads (each with their own
        weights), but the K and V projections only have <code>G</code> heads.
        When attention is computed for query head <code>h</code>, it dot-products
        against the K of group <code>floor(h / group_size)</code> — and reads
        from the same group's V. KV cache shrinks by a factor of{" "}
        <code>H/G</code> with no change to the Q dimension and minimal accuracy
        loss in practice.
      </p>

      <h2>Qwen3.5-0.8B specifics</h2>
      <p>
        Qwen3.5-0.8B uses <code>num_heads = {QWEN_NUM_HEADS}</code>,{" "}
        <code>num_kv_heads = {QWEN_NUM_KV_HEADS}</code>,{" "}
        <code>head_dim = {QWEN_HEAD_DIM}</code>. Each K/V head is shared across{" "}
        <code>{QWEN_NUM_HEADS / QWEN_NUM_KV_HEADS}</code> query heads, so its
        KV cache is <strong>{QWEN_NUM_HEADS / QWEN_NUM_KV_HEADS}× smaller</strong>{" "}
        than the equivalent full-MHA model. Larger Qwen3.5 variants keep the
        same <code>group_size = 4</code> ratio.
      </p>
      <p>
        Layers in Qwen3.5 alternate: most are <em>linear-attention</em> (a
        recurrent variant outside this chapter's scope), and every fourth layer
        is the classic <em>full-attention</em> layer with softmax scores you
        can inspect. The layer selector below restricts to the latter.
      </p>

      <h2>The trade-off, and where MQA fits</h2>
      <p>
        GQA gives up some expressivity — query heads in the same group cannot
        attend to different keys, because they share K. They <em>can</em>{" "}
        still learn different attention patterns over those shared keys via
        their distinct Q projections (you'll see this in the side-by-side
        below). The memory win is large, the quality loss is small, and the
        decode speedup from a smaller cache is real. Mid-size open models like
        Llama 3 and Qwen3.5 all use GQA.
      </p>
      <p>
        Push GQA to the extreme — <code>num_kv_heads = 1</code> — and you get{" "}
        <strong>Multi-Query Attention (MQA)</strong>: every query head shares
        one K/V. Earlier inference-focused models (PaLM, Falcon) used MQA. The
        consensus today is that GQA with a moderate group size (typically 4–8)
        recovers most of MHA's quality at most of MQA's cache savings.
      </p>

      <p className="mt-6 text-muted-foreground">
        Try the widget on the right. The lane diagram shows which query heads
        share which K/V head. The two-up heatmap compares two query heads in
        the same group — same K, different Q, so different attention patterns
        over the same keys.
      </p>
    </Prose>
  );
}

export type MultiheadGqaDemoProps = {
  workerRef: React.RefObject<Worker | null>;
  abortRef: React.RefObject<AbortController | null>;
};

type RunStatus =
  | { kind: "ok" }
  | { kind: "mock-no-worker" }
  | { kind: "mock-error"; error: string }
  | { kind: "aborted" }
  | { kind: "empty-prompt" };

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function pickFullAttentionLayers(run: AttentionRun): AttentionLayer[] {
  return run.attention.filter((l) => l.kind === "full" && l.scores.length > 0);
}

function defaultHeadCounts(run: AttentionRun | null): {
  numHeads: number;
  numKvHeads: number;
} {
  if (run) {
    const layer = pickFullAttentionLayers(run)[0];
    if (layer && layer.numHeads > 0 && layer.numKvHeads > 0) {
      return {
        numHeads: layer.numHeads,
        numKvHeads: layer.numKvHeads,
      };
    }
  }
  return { numHeads: QWEN_NUM_HEADS, numKvHeads: QWEN_NUM_KV_HEADS };
}

export function MultiheadGqaDemo({
  workerRef,
  abortRef,
}: MultiheadGqaDemoProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_PROMPT);
  const [run, setRun] = React.useState<AttentionRun | null>(null);
  const [status, setStatus] = React.useState<RunStatus | null>(null);
  const [running, setRunning] = React.useState(false);
  const [selectedLayerIdx, setSelectedLayerIdx] = React.useState(0);
  const [selectedHead, setSelectedHead] = React.useState(0);

  const runAbortRef = React.useRef<AbortController | null>(null);
  const runGenRef = React.useRef(0);

  React.useEffect(
    () => () => {
      runAbortRef.current?.abort();
    },
    [],
  );

  function applyMockFallback(reason: RunStatus, logMessage: string) {
    const mock = makeMockAttentionRun(prompt);
    console.log(logMessage, {
      promptLength: prompt.length,
      seqLen: mock.tokens.length,
    });
    setRun(mock);
    setStatus(reason);
    setSelectedLayerIdx(0);
    setSelectedHead(0);
  }

  async function handleRun() {
    if (prompt.trim().length === 0) {
      setStatus({ kind: "empty-prompt" });
      return;
    }

    const myGen = ++runGenRef.current;
    setRunning(true);
    const worker = workerRef.current;
    if (!worker) {
      applyMockFallback(
        { kind: "mock-no-worker" },
        "[multihead-gqa-demo] using mock AttentionRun (model not loaded)",
      );
      setRunning(false);
      return;
    }

    runAbortRef.current?.abort();
    const ctrl = new AbortController();
    runAbortRef.current = ctrl;

    const appSignal = abortRef.current?.signal;
    if (appSignal?.aborted) {
      ctrl.abort();
    }
    const onAppAbort = () => ctrl.abort();
    if (appSignal && !appSignal.aborted) {
      appSignal.addEventListener("abort", onAppAbort, { once: true });
    }

    try {
      const result = await runForInspector(
        worker,
        {
          prompt,
          attention: true,
          maxNewTokens: 1,
        },
        { signal: ctrl.signal },
      );
      if (runGenRef.current !== myGen) return;
      console.log("[multihead-gqa-demo] runForInspector ok", {
        promptLength: prompt.length,
        attentionLayers: result.attention.length,
      });
      setRun(result);
      setStatus({ kind: "ok" });
      setSelectedLayerIdx(0);
      setSelectedHead(0);
    } catch (err) {
      if (runGenRef.current !== myGen) return;
      if (isAbortError(err)) {
        console.info(
          "[multihead-gqa-demo] runForInspector aborted (worker terminated or superseded)",
        );
        if (appSignal?.aborted) {
          setStatus({ kind: "aborted" });
        }
      } else {
        const message = err instanceof Error ? err.message : String(err);
        console.error(
          "[multihead-gqa-demo] runForInspector failed; falling back to mock",
          err,
        );
        applyMockFallback(
          { kind: "mock-error", error: message },
          "[multihead-gqa-demo] using mock AttentionRun (backend error)",
        );
      }
    } finally {
      if (appSignal) appSignal.removeEventListener("abort", onAppAbort);
      if (runAbortRef.current === ctrl) runAbortRef.current = null;
      if (runGenRef.current === myGen) setRunning(false);
    }
  }

  const fullLayers = React.useMemo(
    () => (run ? pickFullAttentionLayers(run) : []),
    [run],
  );

  React.useEffect(() => {
    if (fullLayers.length === 0) return;
    if (selectedLayerIdx >= fullLayers.length) {
      setSelectedLayerIdx(0);
    }
  }, [fullLayers, selectedLayerIdx]);

  const currentLayer = fullLayers[selectedLayerIdx] ?? null;

  React.useEffect(() => {
    if (!currentLayer) return;
    if (selectedHead >= currentLayer.numHeads) {
      setSelectedHead(0);
    }
  }, [currentLayer, selectedHead]);

  const { numHeads, numKvHeads } = defaultHeadCounts(run);
  const groupSize = Math.max(1, Math.floor(numHeads / Math.max(1, numKvHeads)));
  const selectedGroup = Math.floor(selectedHead / groupSize);
  const partnerHead = React.useMemo(() => {
    const groupStart = selectedGroup * groupSize;
    const groupEnd = Math.min(groupStart + groupSize, numHeads);
    for (let h = groupStart; h < groupEnd; h++) {
      if (h !== selectedHead) return h;
    }
    return selectedHead;
  }, [selectedGroup, groupSize, numHeads, selectedHead]);

  const seqLen = run?.tokens.length ?? 0;
  const decodeSeqLen = Math.max(seqLen, ASSUMED_DECODE_SEQ_LEN);

  const hasRun = run !== null;
  const showHeatmaps =
    hasRun && currentLayer !== null && currentLayer.scores.length > 0;

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <label
          htmlFor="multihead-gqa-demo-input"
          className="text-xs uppercase tracking-wider text-muted-foreground"
        >
          Prompt
        </label>
        <Textarea
          id="multihead-gqa-demo-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          className="font-mono text-sm"
          placeholder={DEFAULT_PROMPT}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run"}
          </Button>
          <span className="text-xs text-muted-foreground">
            One forward pass, captures attention scores for every full
            layer/head.
          </span>
        </div>
      </div>

      {status?.kind === "mock-no-worker" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Showing example data — load the model from the chat tab first to see
          real attention scores from Qwen3.5-0.8B.
        </div>
      ) : null}
      {status?.kind === "mock-error" ? (
        <div
          role="alert"
          className="rounded-md border border-destructive/60 bg-destructive/10 px-3 py-2 text-xs text-destructive"
        >
          <strong>Inspector run failed.</strong> Showing example data instead.{" "}
          {status.error}
        </div>
      ) : null}
      {status?.kind === "aborted" ? (
        <div
          role="status"
          className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
        >
          Request cancelled — model was reloaded. Click <strong>Run</strong>{" "}
          again to retry.
        </div>
      ) : null}
      {status?.kind === "empty-prompt" ? (
        <div
          role="alert"
          className="rounded-md border border-dashed border-amber-500/60 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300"
        >
          Enter a prompt to capture multi-head attention scores.
        </div>
      ) : null}

      <GroupLaneDiagram
        numHeads={numHeads}
        numKvHeads={numKvHeads}
        selectedHead={selectedHead}
        onSelectHead={(h) => setSelectedHead(h)}
      />

      <KvCacheTicker
        numHeads={numHeads}
        numKvHeads={numKvHeads}
        headDim={QWEN_HEAD_DIM}
        seqLen={decodeSeqLen}
        seqLenSource={seqLen > 0 ? "prompt" : "assumed"}
      />

      {hasRun ? (
        <>
          {fullLayers.length > 0 ? (
            <div className="flex flex-wrap items-center gap-2">
              <label
                htmlFor="multihead-gqa-demo-layer"
                className="text-xs uppercase tracking-wider text-muted-foreground"
              >
                Full layer
              </label>
              <Select
                value={`${selectedLayerIdx}`}
                onValueChange={(v) => setSelectedLayerIdx(Number(v))}
              >
                <SelectTrigger
                  id="multihead-gqa-demo-layer"
                  size="sm"
                  className="min-w-[10rem]"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {fullLayers.map((layer, idx) => (
                    <SelectItem key={layer.layerIndex} value={`${idx}`}>
                      {`Layer ${layer.layerIndex} (full)`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <label
                htmlFor="multihead-gqa-demo-head"
                className="ml-2 text-xs uppercase tracking-wider text-muted-foreground"
              >
                Head
              </label>
              <Select
                value={`${selectedHead}`}
                onValueChange={(v) => setSelectedHead(Number(v))}
              >
                <SelectTrigger
                  id="multihead-gqa-demo-head"
                  size="sm"
                  className="min-w-[8rem]"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: numHeads }).map((_, h) => (
                    <SelectItem key={h} value={`${h}`}>
                      {`Q${h} → KV${Math.floor(h / groupSize)}`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-xs text-muted-foreground">
                {run?.modelMeta.name ?? "—"}
              </span>
            </div>
          ) : (
            <div
              role="status"
              className="rounded-md border border-dashed border-muted-foreground/40 bg-muted/40 px-3 py-2 text-xs text-muted-foreground"
            >
              No full-attention layers in this run. The mock data set does not
              cover GQA per-layer scores end-to-end — load the real model to
              compare heatmaps within a group.
            </div>
          )}

          {showHeatmaps && currentLayer && partnerHead !== selectedHead ? (
            <GroupHeadCompare
              run={run!}
              layer={currentLayer}
              headA={selectedHead}
              headB={partnerHead}
              groupSize={groupSize}
            />
          ) : null}

          <details className="rounded-md border border-border bg-background">
            <summary className="cursor-pointer px-3 py-2 text-xs uppercase tracking-wider text-muted-foreground">
              Full heatmap (single head, interactive)
            </summary>
            <div className="px-3 pb-3 pt-1">
              <AttentionHeatmap run={run!} />
            </div>
          </details>
        </>
      ) : (
        <div className="rounded-md border border-dashed border-border p-6 text-sm text-muted-foreground">
          Click <strong>Run</strong> to capture attention scores and compare
          query heads within a K/V group.
        </div>
      )}
    </div>
  );
}

function GroupLaneDiagram({
  numHeads,
  numKvHeads,
  selectedHead,
  onSelectHead,
}: {
  numHeads: number;
  numKvHeads: number;
  selectedHead: number;
  onSelectHead: (head: number) => void;
}) {
  const groupSize = Math.max(1, Math.floor(numHeads / Math.max(1, numKvHeads)));
  const selectedGroup = Math.floor(selectedHead / groupSize);

  const width = 560;
  const height = 180;
  const padX = 12;
  const qRowY = 36;
  const kRowY = 132;
  const qSlot = (width - 2 * padX) / numHeads;
  const kSlot = (width - 2 * padX) / numKvHeads;
  const qChipW = Math.max(20, qSlot - 6);
  const qChipH = 24;
  const kChipW = Math.max(40, kSlot - 8);
  const kChipH = 28;

  function qChipX(h: number) {
    return padX + h * qSlot + (qSlot - qChipW) / 2;
  }
  function kChipX(g: number) {
    return padX + g * kSlot + (kSlot - kChipW) / 2;
  }

  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          GQA lane map — {numHeads} query heads, {numKvHeads} KV heads, group
          size {groupSize}
        </div>
        <div className="text-[11px] text-muted-foreground">
          Each color is one group. Click a query chip to focus it.
        </div>
      </div>
      <svg
        role="img"
        aria-label={`GQA lane diagram: ${numHeads} query heads grouped into ${numKvHeads} KV heads`}
        viewBox={`0 0 ${width} ${height}`}
        className="block h-auto w-full max-w-[600px]"
      >
        <text
          x={padX}
          y={qRowY - 14}
          className="fill-muted-foreground"
          style={{ fontSize: 10, letterSpacing: "0.05em" }}
        >
          QUERY HEADS (EACH HAS ITS OWN Wq)
        </text>
        <text
          x={padX}
          y={kRowY + kChipH + 20}
          className="fill-muted-foreground"
          style={{ fontSize: 10, letterSpacing: "0.05em" }}
        >
          K/V HEADS (SHARED ACROSS THE GROUP)
        </text>

        {Array.from({ length: numHeads }).map((_, h) => {
          const group = Math.floor(h / groupSize);
          const isSelected = h === selectedHead;
          const inSelectedGroup = group === selectedGroup;
          const dim = !inSelectedGroup;
          const qx = qChipX(h);
          const kx = kChipX(group);
          return (
            <line
              key={`line-${h}`}
              x1={qx + qChipW / 2}
              y1={qRowY + qChipH}
              x2={kx + kChipW / 2}
              y2={kRowY}
              className={
                dim
                  ? "stroke-muted-foreground/20"
                  : isSelected
                    ? "stroke-foreground/80"
                    : "stroke-foreground/40"
              }
              strokeWidth={isSelected ? 1.6 : 1}
            />
          );
        })}

        {Array.from({ length: numKvHeads }).map((_, g) => {
          const color = GROUP_COLORS[g % GROUP_COLORS.length]!;
          const isSelectedGroup = g === selectedGroup;
          const opacity = isSelectedGroup ? 1 : 0.55;
          return (
            <g key={`kv-${g}`} opacity={opacity}>
              <rect
                x={kChipX(g)}
                y={kRowY}
                width={kChipW}
                height={kChipH}
                rx={6}
                fill={color.fill}
              />
              <text
                x={kChipX(g) + kChipW / 2}
                y={kRowY + kChipH / 2 + 4}
                textAnchor="middle"
                fill={color.text}
                style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 11 }}
              >
                {`K/V ${g}`}
              </text>
            </g>
          );
        })}

        {Array.from({ length: numHeads }).map((_, h) => {
          const group = Math.floor(h / groupSize);
          const color = GROUP_COLORS[group % GROUP_COLORS.length]!;
          const isSelected = h === selectedHead;
          const inSelectedGroup = group === selectedGroup;
          const opacity = inSelectedGroup ? 1 : 0.45;
          const activate = () => onSelectHead(h);
          return (
            <g
              key={`q-${h}`}
              opacity={opacity}
              role="button"
              tabIndex={0}
              aria-label={`Select query head ${h}`}
              aria-pressed={isSelected}
              onClick={activate}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  activate();
                }
              }}
              style={{ cursor: "pointer", outline: "none" }}
            >
              <rect
                x={qChipX(h)}
                y={qRowY}
                width={qChipW}
                height={qChipH}
                rx={5}
                fill={color.fill}
              />
              {isSelected ? (
                <rect
                  x={qChipX(h) - 2}
                  y={qRowY - 2}
                  width={qChipW + 4}
                  height={qChipH + 4}
                  rx={6}
                  fill="none"
                  className="stroke-foreground"
                  strokeWidth={1.6}
                />
              ) : null}
              <text
                x={qChipX(h) + qChipW / 2}
                y={qRowY + qChipH / 2 + 4}
                textAnchor="middle"
                fill={color.text}
                pointerEvents="none"
                style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 10 }}
              >
                {`Q${h}`}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="text-[11px] text-muted-foreground">
        Selected: query head <span className="font-mono">Q{selectedHead}</span>{" "}
        — group{" "}
        <span className="font-mono">{selectedGroup}</span> (KV head{" "}
        <span className="font-mono">K/V {selectedGroup}</span>), shared with{" "}
        {groupSize - 1} other query head{groupSize - 1 === 1 ? "" : "s"} in the
        same group.
      </div>
    </div>
  );
}

function KvCacheTicker({
  numHeads,
  numKvHeads,
  headDim,
  seqLen,
  seqLenSource,
}: {
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  seqLen: number;
  seqLenSource: "prompt" | "assumed";
}) {
  const mha = 2 * numHeads * headDim * seqLen;
  const gqa = 2 * numKvHeads * headDim * seqLen;
  const ratio = gqa > 0 ? mha / gqa : 0;
  return (
    <div className="grid grid-cols-1 gap-2 rounded-md border border-border bg-background p-3 sm:grid-cols-3">
      <Stat
        label="MHA cache / layer"
        value={`${formatCount(mha)} floats`}
        sub={`2 · ${numHeads} · ${headDim} · ${formatCount(seqLen)}`}
      />
      <Stat
        label="GQA cache / layer"
        value={`${formatCount(gqa)} floats`}
        sub={`2 · ${numKvHeads} · ${headDim} · ${formatCount(seqLen)}`}
        accent
      />
      <Stat
        label="Savings"
        value={`${ratio.toFixed(1)}× smaller`}
        sub={
          seqLenSource === "assumed"
            ? `assumed seq_len=${formatCount(seqLen)}`
            : `seq_len=${formatCount(seqLen)} from prompt`
        }
      />
    </div>
  );
}

function Stat({
  label,
  value,
  sub,
  accent = false,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-md border p-2",
        accent
          ? "border-primary/50 bg-primary/5"
          : "border-border bg-muted/30",
      ].join(" ")}
    >
      <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-sm text-foreground">{value}</div>
      <div className="text-[11px] text-muted-foreground">{sub}</div>
    </div>
  );
}

function formatCount(n: number): string {
  if (!Number.isFinite(n)) return "—";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

function GroupHeadCompare({
  run,
  layer,
  headA,
  headB,
  groupSize,
}: {
  run: AttentionRun;
  layer: AttentionLayer;
  headA: number;
  headB: number;
  groupSize: number;
}) {
  const groupIdx = Math.floor(headA / groupSize);
  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Same K/V, different attention — group {groupIdx}
        </div>
        <div className="text-[11px] text-muted-foreground">
          Both heads attend over the same K/V {groupIdx}; their distinct Wq
          shape the patterns.
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <SmallHeatmap
          run={run}
          layer={layer}
          head={headA}
          accent
          caption={`Q${headA}`}
        />
        <SmallHeatmap
          run={run}
          layer={layer}
          head={headB}
          accent={false}
          caption={`Q${headB}`}
        />
      </div>
    </div>
  );
}

function SmallHeatmap({
  run,
  layer,
  head,
  accent,
  caption,
}: {
  run: AttentionRun;
  layer: AttentionLayer;
  head: number;
  accent: boolean;
  caption: string;
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const seqLen = run.tokens.length;
  const maxBoardPx = 240;
  const cellSize = Math.max(
    6,
    Math.min(20, Math.floor(maxBoardPx / Math.max(1, seqLen))),
  );
  const board = cellSize * seqLen;

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = board * dpr;
    canvas.height = board * dpr;
    canvas.style.width = `${board}px`;
    canvas.style.height = `${board}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, board, board);

    if (layer.scores.length === 0) {
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.font = "11px var(--font-mono, monospace)";
      ctx.fillText("No scores available.", 8, 18);
      return;
    }
    drawAttentionHeatmap(ctx, layer, head, cellSize, {
      colorFn: accent ? colorAccent : colorNeutral,
    });
  }, [run, layer, head, seqLen, cellSize, board, accent]);

  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
        <span className="font-mono">{caption}</span>
        <span>
          {seqLen}×{seqLen}
        </span>
      </div>
      <canvas
        ref={canvasRef}
        className="block rounded-sm border border-border"
        style={{ width: board, height: board }}
        role="img"
        aria-label={`Attention heatmap head ${head}`}
      />
    </div>
  );
}

function colorAccent(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const l = 14 + t * 56;
  return `hsl(210 70% ${l.toFixed(0)}%)`;
}

function colorNeutral(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  const l = 14 + t * 56;
  return `hsl(310 70% ${l.toFixed(0)}%)`;
}
