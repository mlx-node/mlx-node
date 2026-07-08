import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * MtpModuleDiagram — chapter 14 "Architecture", the Multi-Token-Prediction
 * (MTP) deep-dive. A STATIC top-to-bottom data-flow diagram of one Qwen3.5 MTP
 * module's internals — no animation, no worker, no model, no WASM. Pure SVG +
 * React, so it server-renders for crawlers (createRoot replaces it on boot) and
 * is robust while the model downloads.
 *
 * Ground-truth verified against the mlx-node engine config
 * (crates/mlx-core/src/models/qwen3_5/config.rs: n_mtp_layers from
 * mtp_num_hidden_layers) AND vLLM's Qwen3_5MultiTokenPredictor.forward
 * (vllm/model_executor/models/qwen3_5_mtp.py, mirrored by qwen3_next_mtp.py):
 *
 *   inputs_embeds = pre_fc_norm_embedding(inputs_embeds)   # shared embed matrix
 *   hidden_states = pre_fc_norm_hidden(hidden_states)      # main model's final h
 *   hidden_states = cat([inputs_embeds, hidden_states])    # 2048 = 2 × 1024
 *   hidden_states = fc(hidden_states)                      # Linear 2048→1024, no bias
 *   hidden_states = layers[0](...)                         # ONE full_attention layer
 *   hidden_states = norm(hidden_states)                    # final RMSNorm
 *   logits        = lm_head(hidden_states)                 # tied weights → draft token
 *
 * The single decoder layer is hard-coded layer_type="full_attention"
 * (qwen3_5_mtp.py L102-104) — it is never GatedDeltaNet and never touches the
 * recurrent state. Embedding + LM head are tied to the main model
 * (tie_word_embeddings → lm_head = embed_tokens), so the head carries no
 * vocab-sized weights of its own; both are drawn dashed/muted to convey
 * "borrowed, not its own weights".
 */

const COPY = {
  en: {
    heading: 'Inside one MTP module',
    ariaLabel:
      "Top-to-bottom data-flow diagram of one Qwen3.5 Multi-Token-Prediction module. Two inputs at the top — the main model's last hidden state and the embedding of the just-emitted token from the shared embedding matrix — each pass through an RMSNorm, are concatenated into a 2048-dim vector, projected by a bias-free Linear back to 1024, run through a single full-attention decoder layer, normalized, and read out by the tied LM head into a draft token.",
    // Two top inputs
    inHiddenTitle: 'last hidden state  h',
    inHiddenDim: '1024-d',
    inHiddenNote: "the main model's final hidden for this position",
    inEmbedTitle: 'token just emitted → embedding',
    inEmbedDim: '1024-d',
    inEmbedNote: "uses the MAIN model's embedding matrix",
    // Steps
    normHidden: 'RMSNorm',
    normHiddenTag: 'pre_fc_norm_hidden',
    normEmbed: 'RMSNorm',
    normEmbedTag: 'pre_fc_norm_embedding',
    concat: 'concat([emb_norm, hidden_norm])',
    concatDim: '2048 = 2 × 1024',
    fc: 'fc',
    fcDesc: 'Linear 2048 → 1024',
    fcTag: 'no bias',
    layerTitle: 'one full-attention decoder layer',
    layerBody: 'self-attention + gated MLP',
    layerNote: 'a FULL-attention layer — never GatedDeltaNet; it does not touch the recurrent state',
    normFinal: 'RMSNorm',
    normFinalTag: 'final',
    head: 'shared LM head (tied weights)',
    output: 'draft token',
    outputNote: 'the one AFTER next',
    legend: 'Dashed boxes are shared with the main model — no new weights of their own.',
    // Captions
    caption:
      "Shares the main model's embedding + LM head, so the head is tiny — one transformer layer, three norms, one projection.",
    config: 'config.json:  mtp_num_hidden_layers: 1  ·  mtp_use_dedicated_embeddings: false',
  },
  zh: {
    heading: '一个 MTP 模块的内部',
    ariaLabel:
      'Qwen3.5 Multi-Token-Prediction 模块的自上而下数据流示意图。顶部有两个输入——主模型的 last hidden state，以及刚生成的 token 经共享 embedding 矩阵得到的 embedding——各自经过一个 RMSNorm，拼接为 2048 维向量，由无 bias 的 Linear 投回 1024 维，穿过单个 full-attention 解码层，归一化后由共享的 LM head 读出为一个草稿 token。',
    inHiddenTitle: 'last hidden state  h',
    inHiddenDim: '1024-d',
    inHiddenNote: '主模型在当前位置的最终 hidden',
    inEmbedTitle: '刚生成的 token → embedding',
    inEmbedDim: '1024-d',
    inEmbedNote: '用的是主模型的 embedding 矩阵',
    normHidden: 'RMSNorm',
    normHiddenTag: 'pre_fc_norm_hidden',
    normEmbed: 'RMSNorm',
    normEmbedTag: 'pre_fc_norm_embedding',
    concat: 'concat([emb_norm, hidden_norm])',
    concatDim: '2048 = 2 × 1024',
    fc: 'fc',
    fcDesc: 'Linear 2048 → 1024',
    fcTag: '无 bias',
    layerTitle: '一个 full-attention 解码层',
    layerBody: 'self-attention + 门控 MLP',
    layerNote: '一个 FULL-attention 层——绝不是 GatedDeltaNet；它不触碰循环状态',
    normFinal: 'RMSNorm',
    normFinalTag: '最终',
    head: '共享 LM head（权重绑定）',
    output: '草稿 token',
    outputNote: '即「下一个之后」的那个',
    legend: '虚线框与主模型共享——本身没有新增权重。',
    caption: '复用主模型的 embedding 与 LM head，所以这个头很小——一个 transformer 层、三个 norm、一个投影。',
    config: 'config.json：  mtp_num_hidden_layers: 1  ·  mtp_use_dedicated_embeddings: false',
  },
} as const;

// ── SVG geometry. A single centred column; the two inputs sit side-by-side at
// the top, each feed a norm, the two norm streams merge into the concat box,
// and from there a single column flows straight down. ─────────────────────────
const VB_W = 640;
const VB_H = 690;

const CX = VB_W / 2; // centre line the trunk runs down
const LX = 158; // centre x of the left (embedding) stream
const RX = 482; // centre x of the right (hidden) stream
const COL_W = 244; // width of a single-column box on the trunk
const SIDE_W = 244; // width of a side (input / norm) box

export function MtpModuleDiagram() {
  const copy = COPY[useLocale()];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.ariaLabel}>
        {/* ── Connectors (under the boxes) ── */}
        {/* embedding input → its norm */}
        <Arrow from={[LX, 84]} to={[LX, 100]} />
        {/* hidden input → its norm */}
        <Arrow from={[RX, 84]} to={[RX, 100]} />
        {/* both norms → concat (elbows that merge into the trunk) */}
        <Elbow from={[LX, 166]} toX={CX} toY={196} />
        <Elbow from={[RX, 166]} toX={CX} toY={196} />
        {/* concat → fc → layer → norm → head → output (straight trunk) */}
        <Arrow from={[CX, 262]} to={[CX, 286]} />
        <Arrow from={[CX, 358]} to={[CX, 382]} />
        <Arrow from={[CX, 478]} to={[CX, 502]} />
        <Arrow from={[CX, 568]} to={[CX, 592]} />
        <Arrow from={[CX, 632]} to={[CX, 654]} />

        {/* ── TWO INPUTS (top) ── */}
        {/* embedding — SHARED (dashed/muted) */}
        <Box x={LX - SIDE_W / 2} y={20} w={SIDE_W} h={64} shared>
          <CenterText x={LX} y={40} kind="title">
            {copy.inEmbedTitle}
          </CenterText>
          <CenterText x={LX} y={58} kind="mono-strong">
            {copy.inEmbedDim}
          </CenterText>
          <CenterText x={LX} y={75} kind="muted">
            {copy.inEmbedNote}
          </CenterText>
        </Box>

        {/* hidden state */}
        <Box x={RX - SIDE_W / 2} y={20} w={SIDE_W} h={64}>
          <CenterText x={RX} y={40} kind="title">
            {copy.inHiddenTitle}
          </CenterText>
          <CenterText x={RX} y={58} kind="mono-strong">
            {copy.inHiddenDim}
          </CenterText>
          <CenterText x={RX} y={75} kind="muted">
            {copy.inHiddenNote}
          </CenterText>
        </Box>

        {/* ── Step 1 & 2: the two pre-fc RMSNorms ── */}
        <Box x={LX - SIDE_W / 2} y={100} w={SIDE_W} h={50} accent>
          <CenterText x={LX} y={124} kind="title">
            {copy.normEmbed}
          </CenterText>
          <CenterText x={LX} y={141} kind="mono">
            {copy.normEmbedTag}
          </CenterText>
        </Box>
        <Box x={RX - SIDE_W / 2} y={100} w={SIDE_W} h={50} accent>
          <CenterText x={RX} y={124} kind="title">
            {copy.normHidden}
          </CenterText>
          <CenterText x={RX} y={141} kind="mono">
            {copy.normHiddenTag}
          </CenterText>
        </Box>

        {/* ── Step 3: concat → 2048-d ── */}
        <Box x={CX - COL_W / 2} y={196} w={COL_W} h={66}>
          <CenterText x={CX} y={222} kind="mono-strong">
            {copy.concat}
          </CenterText>
          <CenterText x={CX} y={246} kind="mono-dim">
            {copy.concatDim}
          </CenterText>
        </Box>

        {/* ── Step 4: fc Linear 2048 → 1024 (no bias) ── */}
        <Box x={CX - COL_W / 2} y={286} w={COL_W} h={72} accent>
          <CenterText x={CX} y={312} kind="title">
            <tspan className="font-mono">{copy.fc}</tspan> · {copy.fcDesc}
          </CenterText>
          <CenterText x={CX} y={334} kind="mono-dim">
            {copy.fcTag}
          </CenterText>
        </Box>

        {/* ── Step 5: ONE full-attention decoder layer ── */}
        <Box x={CX - COL_W / 2} y={382} w={COL_W} h={96} primary>
          <CenterText x={CX} y={408} kind="title-strong">
            {copy.layerTitle}
          </CenterText>
          <CenterText x={CX} y={430} kind="mono">
            {copy.layerBody}
          </CenterText>
        </Box>
        {/* annotation beside the layer */}
        <Note x={CX + COL_W / 2 + 12} y={382} w={VB_W - (CX + COL_W / 2 + 12) - 8} h={96}>
          {copy.layerNote}
        </Note>

        {/* ── Step 6: final RMSNorm ── */}
        <Box x={CX - COL_W / 2} y={502} w={COL_W} h={56} accent>
          <CenterText x={CX} y={528} kind="title">
            {copy.normFinal}
          </CenterText>
          <CenterText x={CX} y={545} kind="mono">
            {copy.normFinalTag}
          </CenterText>
        </Box>

        {/* ── Step 7: shared LM head (dashed/muted) → output ── */}
        <Box x={CX - COL_W / 2} y={592} w={COL_W} h={40} shared>
          <CenterText x={CX} y={616} kind="title">
            {copy.head}
          </CenterText>
        </Box>

        {/* output token */}
        <Box x={CX - COL_W / 2} y={654} w={COL_W} h={28} primary>
          <CenterText x={CX} y={672} kind="output">
            {copy.output} · {copy.outputNote}
          </CenterText>
        </Box>
      </svg>

      {/* Captions */}
      <p className="text-[13px] text-foreground/90">{copy.caption}</p>
      <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
        <span
          aria-hidden
          className="inline-block h-3.5 w-6 shrink-0 rounded-sm border border-dashed"
          style={{ borderColor: 'var(--muted-foreground)', background: 'var(--muted)' }}
        />
        <span>{copy.legend}</span>
      </div>
      <p className="font-mono text-[11px] text-muted-foreground/80">{copy.config}</p>
    </div>
  );
}

type Pt = [number, number];

/** A vertical-ish connector with a small downward arrowhead at its destination. */
function Arrow({ from, to }: { from: Pt; to: Pt }) {
  return (
    <g>
      <line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} style={{ stroke: 'var(--border)' }} strokeWidth={1.75} />
      <polygon
        points={`${to[0]},${to[1]} ${to[0] - 5},${to[1] - 8} ${to[0] + 5},${to[1] - 8}`}
        style={{ fill: 'var(--border)' }}
      />
    </g>
  );
}

/** A merging connector: down from a side stream, then into the centre trunk, arrowhead at the trunk. */
function Elbow({ from, toX, toY }: { from: Pt; toX: number; toY: number }) {
  const midY = (from[1] + toY) / 2;
  const d = `M ${from[0]} ${from[1]} L ${from[0]} ${midY} L ${toX} ${midY} L ${toX} ${toY}`;
  return (
    <g>
      <path d={d} fill="none" style={{ stroke: 'var(--border)' }} strokeWidth={1.75} />
      <polygon points={`${toX},${toY} ${toX - 5},${toY - 8} ${toX + 5},${toY - 8}`} style={{ fill: 'var(--border)' }} />
    </g>
  );
}

/** A box on the flow. `shared` → dashed/muted (borrowed weights). `primary`/`accent` → highlighted. */
function Box({
  x,
  y,
  w,
  h,
  shared,
  primary,
  accent,
  children,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  shared?: boolean;
  primary?: boolean;
  accent?: boolean;
  children: React.ReactNode;
}) {
  const stroke = shared ? 'var(--muted-foreground)' : primary || accent ? 'var(--primary)' : 'var(--border)';
  const fill = primary ? 'var(--primary)' : shared ? 'var(--muted)' : 'var(--card)';
  const fillOpacity = primary ? 0.08 : 1;
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={10}
        style={{ fill, fillOpacity, stroke }}
        strokeWidth={primary ? 2 : 1.5}
        strokeDasharray={shared ? '5 4' : undefined}
      />
      {children}
    </g>
  );
}

type TextKind = 'title' | 'title-strong' | 'mono' | 'mono-strong' | 'mono-dim' | 'muted' | 'output';

/** Centre-anchored label. Mono variants use the design-token mono font; never below 11px. */
function CenterText({ x, y, kind, children }: { x: number; y: number; kind: TextKind; children: React.ReactNode }) {
  const map: Record<TextKind, { cls: string; fill: string }> = {
    title: { cls: 'text-[13px] font-semibold', fill: 'var(--foreground)' },
    'title-strong': { cls: 'text-[14px] font-semibold', fill: 'var(--foreground)' },
    mono: { cls: 'font-mono text-[11px]', fill: 'var(--muted-foreground)' },
    'mono-strong': { cls: 'font-mono text-[12px] font-semibold', fill: 'var(--foreground)' },
    'mono-dim': { cls: 'font-mono text-[11px]', fill: 'var(--muted-foreground)' },
    muted: { cls: 'text-[11px]', fill: 'var(--muted-foreground)' },
    output: { cls: 'text-[12px] font-semibold', fill: 'var(--primary)' },
  };
  const { cls, fill } = map[kind];
  return (
    <text x={x} y={y} textAnchor="middle" style={{ fill }} className={cls}>
      {children}
    </text>
  );
}

/** A side annotation: foreignObject so prose can wrap. */
function Note({ x, y, w, h, children }: { x: number; y: number; w: number; h: number; children: React.ReactNode }) {
  return (
    <foreignObject x={x} y={y} width={w} height={h}>
      <div className="flex h-full items-center text-[11px] leading-snug text-muted-foreground">{children}</div>
    </foreignObject>
  );
}
