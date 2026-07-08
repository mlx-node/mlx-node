import { cn } from '@/lib/utils';
import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 15 (post-training) supplement — what "preference tuning" actually
 * does, in two beats. Pure presentational: only React + `cn`, no model / worker
 * / WASM / network. The numbers are illustrative, not from a model.
 *
 * BEAT 1 — the preference loop (prompt → A/B → human picks → nudge).
 *   A fixed prompt and two candidate responses. The learner clicks which one a
 *   human prefers (👤). Two complementary probability bars show the model's
 *   P(A) and P(B). "Apply preference update" nudges the WINNER's bar UP and the
 *   LOSER's DOWN by one STEP; Reset returns to the even 50/50 start. The whole
 *   point: an update raises the chosen response's probability and lowers the
 *   rejected one — repeated over thousands of pairs, that is all preference
 *   tuning does.
 *
 *   We store ONE number, `pA` ∈ [CLAMP_MIN, CLAMP_MAX]; P(B) = 1 − P(A), so the
 *   two bars are always complementary by construction. The update is a discrete,
 *   user-initiated button press (fine under prefers-reduced-motion); there is no
 *   auto-animation. Switching which response is preferred mid-training does NOT
 *   reset the bars — the next updates simply push the other way from wherever
 *   the split currently is, so the learner can watch the trend reverse.
 *
 * BEAT 2 — RLHF vs DPO (the structural one-liner, made visual).
 *   A toggle redraws the pipeline between the prompt/pair and the model update:
 *     • RLHF: pair → a reward-model box (scores responses) → an RL update. The
 *       reward model is PRESENT and on the path.
 *     • DPO: pair → a direct arrow to the update. The reward-model box is greyed
 *       out and bypassed. DPO optimizes the preference pairs directly.
 *   This visualizes the chapter's existing prose one-liner (reinforcing it).
 */

type Side = 'A' | 'B';
type Method = 'RLHF' | 'DPO';

// The starting (roughly even) split and how far each click nudges P(A).
const P_START = 0.5;
const STEP = 0.09;
// Keep the divergence visible but bounded — never a literal 0% or 100%.
const CLAMP_MIN = 0.02;
const CLAMP_MAX = 0.98;

// Bar colors (winner / loser read clearly in the dark theme).
const A_BG = 'oklch(0.72 0.15 150 / 0.5)'; // green-ish
const B_BG = 'oklch(0.7 0.13 60 / 0.5)'; // amber-ish

// Per-locale copy. The prompt and the two illustrative one-liners localize too:
// the learner has to READ and judge them, and the zh chapter doesn't quote them
// in English. A is the helpful answer; B is the evasive one — plausibly
// comparable, so "picking" the better one feels real.
const COPY = {
  en: {
    header: 'Preference tuning — pick a winner, nudge',
    subnote: (
      <>
        A human prefers one response; the update raises <span className="font-mono">P(chosen)</span>
      </>
    ),
    promptLabel: 'Prompt: ',
    prompt: 'Explain recursion to a 5-year-old.',
    responseA:
      "It's like a tiny robot that, to do a big job, makes a smaller copy of itself to do a smaller piece — until the piece is so small it just does it.",
    responseB: "Recursion is when a function is recursive. It's a standard concept; you'll understand it eventually.",
    chosen: 'chosen',
    pickAria: (side: Side) => `Human prefers response ${side}`,
    responseLabel: (side: Side) => `Response ${side}`,
    preferredChip: '👤 preferred',
    humanPrefers: 'Human prefers:',
    aBetter: 'A is better',
    bBetter: 'B is better',
    probLabel: (side: Side) => `P(${side}) — model probability of response ${side}`,
    applyUpdate: 'Apply preference update →',
    reset: 'Reset',
    updatesCount: (n: number) => `${n} updates`,
    chosenReadout: (side: Side) => `P(chosen) — ${side}`,
    rejectedReadout: (side: Side) => `P(rejected) — ${side}`,
    takeawayCeiling: (winner: Side, ceil: string) =>
      `Response ${winner} is now near the ${ceil}% ceiling — the update raised the chosen response and lowered the rejected one. That's all preference tuning does, repeated over thousands of pairs.`,
    takeawayStart: (winner: Side, loser: Side) =>
      `Click "apply preference update" to nudge the chosen response (${winner}) up and the rejected one (${loser}) down.`,
    takeawayProgress: (updates: number, winner: Side, loser: Side) =>
      `${updates} update${updates === 1 ? '' : 's'}: P(${winner}) is rising, P(${loser}) is falling — the update raises the chosen response's probability and lowers the rejected one.`,
    liveSummary: (preferred: Side, pa: string, pb: string, updates: number, method: Method) =>
      `Human prefers response ${preferred}. ` +
      `Model probability of A is ${pa} percent, of B is ${pb} percent, ` +
      `after ${updates} preference update${updates === 1 ? '' : 's'}. Method: ${method}.`,
    howComputed: 'How the update is computed',
    methodAria: 'Preference-tuning method',
    svgAria: (method: Method, preferred: Side, pa: string, pb: string, rlhf: boolean) =>
      `Preference-tuning pipeline, ${method} mode. ` +
      `A fixed prompt with two candidate responses (the human currently prefers response ${preferred}, ` +
      `model probabilities P(A) ${pa} percent and P(B) ${pb} percent) ` +
      (rlhf
        ? 'flows into a separate reward model that scores the responses, then a reinforcement-learning step updates the model.'
        : 'flows directly into the model update; the reward model is bypassed and greyed out.'),
    svgPairTitle: 'prompt + pair',
    svgPairSub: (preferred: Side) => `A vs B, human picked ${preferred}`,
    svgReward: 'reward model',
    svgScores: 'scores responses',
    svgSkipped: 'skipped',
    svgUpdate: 'model update',
    svgRlStep: 'RL step',
    svgPrefLoss: 'preference loss',
    svgDirect: 'direct — skips the reward model',
    rlhfExplain: (
      <>
        <strong>RLHF</strong> trains a <strong>separate reward model</strong> from the human comparisons, then
        uses reinforcement learning to optimize the model against that reward.
      </>
    ),
    dpoExplain: (
      <>
        <strong>DPO</strong> optimizes the preference pairs <strong>directly</strong> with a preference loss — no
        separate reward model. Same goal as RLHF, one fewer moving part.
      </>
    ),
    footnote: (
      <>
        Illustrative — this is a cartoon. Real preference datasets are thousands of pairs, and the “nudge” is one
        gradient step on a preference loss (DPO) or a reward-model-guided RL step (RLHF); the probabilities here are
        hand-set, not from a model.
      </>
    ),
  },
  zh: {
    header: '偏好微调——选出赢家，轻推一下',
    subnote: (
      <>
        人类偏好其中一条回复；更新会提高 <span className="font-mono">P(chosen)</span>
      </>
    ),
    promptLabel: '提示词：',
    prompt: '向一个 5 岁小孩解释递归。',
    responseA:
      '它就像一个小机器人：要完成一件大事，就复制一个更小的自己去做更小的一块——直到那块小到可以直接做完。',
    responseB: '递归就是一个函数是递归的。这是个标准概念；你以后总会懂的。',
    chosen: '偏好',
    pickAria: (side: Side) => `人类偏好回复 ${side}`,
    responseLabel: (side: Side) => `回复 ${side}`,
    preferredChip: '👤 偏好',
    humanPrefers: '人类偏好：',
    aBetter: 'A 更好',
    bBetter: 'B 更好',
    probLabel: (side: Side) => `P(${side})——模型给回复 ${side} 的概率`,
    applyUpdate: '应用偏好更新 →',
    reset: '重置',
    updatesCount: (n: number) => `${n} 次更新`,
    chosenReadout: (side: Side) => `P(偏好) — ${side}`,
    rejectedReadout: (side: Side) => `P(拒绝) — ${side}`,
    takeawayCeiling: (winner: Side, ceil: string) =>
      `回复 ${winner} 已接近 ${ceil}% 的上限——更新提高了被选中的回复、压低了被拒绝的那条。偏好微调做的就是这件事，在成千上万个回复对上重复。`,
    takeawayStart: (winner: Side, loser: Side) =>
      `点击“应用偏好更新”，把被选中的回复（${winner}）往上推、把被拒绝的（${loser}）往下压。`,
    takeawayProgress: (updates: number, winner: Side, loser: Side) =>
      `${updates} 次更新：P(${winner}) 在上升，P(${loser}) 在下降——更新提高被选中回复的概率、降低被拒绝回复的概率。`,
    liveSummary: (preferred: Side, pa: string, pb: string, updates: number, method: Method) =>
      `人类偏好回复 ${preferred}。模型给 A 的概率为 ${pa}%，给 B 的为 ${pb}%，已应用 ${updates} 次偏好更新。方法：${method}。`,
    howComputed: '更新是如何计算的',
    methodAria: '偏好微调方法',
    svgAria: (method: Method, preferred: Side, pa: string, pb: string, rlhf: boolean) =>
      `偏好微调流水线，${method} 模式。一个固定的提示词和两条候选回复（人类当前偏好回复 ${preferred}，模型概率 P(A) ${pa}%、P(B) ${pb}%）` +
      (rlhf ? '先流入一个独立的、为回复打分的奖励模型，再由一步强化学习更新模型。' : '直接流入模型更新；奖励模型被绕过并置灰。'),
    svgPairTitle: '提示词 + 回复对',
    svgPairSub: (preferred: Side) => `A vs B，人类选了 ${preferred}`,
    svgReward: '奖励模型',
    svgScores: '为回复打分',
    svgSkipped: '被跳过',
    svgUpdate: '模型更新',
    svgRlStep: 'RL 更新',
    svgPrefLoss: '偏好损失',
    svgDirect: '直接——跳过奖励模型',
    rlhfExplain: (
      <>
        <strong>RLHF</strong> 先用人类比较数据训练一个<strong>独立的奖励模型</strong>
        ，再用强化学习去优化主模型在这个奖励上的得分。
      </>
    ),
    dpoExplain: (
      <>
        <strong>DPO</strong> 用一个偏好损失<strong>直接</strong>在偏好对上优化——不需要单独的奖励模型。目标与 RLHF
        相同，少一个活动部件。
      </>
    ),
    footnote: (
      <>
        示意——这是一幅卡通。真实的偏好数据集有成千上万个回复对，而这里的“轻推”对应偏好损失上的一步梯度（DPO）
        或一步由奖励模型引导的 RL 更新（RLHF）；这些概率是手工设定的，并非来自模型。
      </>
    ),
  },
} as const;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

const EASE = 'cubic-bezier(0.4, 0, 0.2, 1)';

// ---- Beat 1: a single probability bar ----
function ProbBar({
  label,
  pct,
  bg,
  isWinner,
  animate,
}: {
  label: string;
  pct: number;
  bg: string;
  isWinner: boolean;
  animate: boolean;
}) {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-foreground/85">
          {label}
          {isWinner ? (
            <span className="ml-1.5 rounded bg-primary/15 px-1 py-0.5 font-mono text-[9px] uppercase tracking-wider text-primary">
              {copy.chosen}
            </span>
          ) : null}
        </span>
        <span className="font-mono text-muted-foreground">{pct.toFixed(0)}%</span>
      </div>
      <div className="flex h-7 w-full overflow-hidden rounded-md border border-border bg-muted/40" aria-hidden="true">
        <div
          className="flex h-full min-w-0 items-center rounded-[3px] px-2"
          style={{
            width: `${pct}%`,
            backgroundColor: bg,
            transition: animate ? `width 320ms ${EASE}` : undefined,
          }}
        >
          {pct > 14 ? (
            <span className="truncate font-mono text-[10px] text-foreground/90">{pct.toFixed(0)}%</span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ---- Beat 1: the "human prefers A / B" pick control ----
function PickButton({
  side,
  active,
  onClick,
  children,
}: {
  side: Side;
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  const copy = COPY[useLocale()];
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={copy.pickAria(side)}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors',
        active
          ? 'border-primary/40 bg-primary/15 text-primary'
          : 'border-border/60 bg-muted/40 text-muted-foreground hover:text-foreground',
      )}
    >
      <span aria-hidden="true">👤</span>
      {children}
    </button>
  );
}

// ---- Beat 2: the RLHF | DPO pipeline diagram (SVG) ----
// Geometry: prompt/pair → [reward model] → model update, left to right.
const PW = 560;
const PH = 132;

function PipelineDiagram({ method, preferred, pA }: { method: Method; preferred: Side; pA: number }) {
  const copy = COPY[useLocale()];
  const rlhf = method === 'RLHF';
  const pB = 1 - pA;

  // Box geometry. Three stages across the strip; the reward model is the middle
  // one — solid + on-path under RLHF, greyed + bypassed under DPO.
  const boxY = 30;
  const boxH = 48;
  const cy = boxY + boxH / 2;
  const pairBox = { x: 16, w: 150 };
  const rewardBox = { x: 205, w: 150 };
  const updateBox = { x: 394, w: 150 };

  const pairR = pairBox.x + pairBox.w;
  const rewardL = rewardBox.x;
  const rewardR = rewardBox.x + rewardBox.w;
  const updateL = updateBox.x;

  const ariaLabel = copy.svgAria(method, preferred, (pA * 100).toFixed(0), (pB * 100).toFixed(0), rlhf);

  return (
    <svg
      role="img"
      aria-label={ariaLabel}
      viewBox={`0 0 ${PW} ${PH}`}
      className="block h-auto w-full rounded-md bg-muted/20"
    >
      <defs>
        <marker id="pp-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 Z" fill="currentColor" />
        </marker>
        <marker
          id="pp-arrow-direct"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6.5"
          markerHeight="6.5"
          orient="auto"
        >
          <path d="M 0 0 L 10 5 L 0 10 Z" fill="oklch(0.72 0.15 150)" />
        </marker>
      </defs>

      {/* prompt + pair */}
      <g>
        <rect
          x={pairBox.x}
          y={boxY}
          width={pairBox.w}
          height={boxH}
          rx={6}
          fill="oklch(0.7 0.13 220 / 0.18)"
          stroke="currentColor"
          strokeOpacity={0.45}
        />
        <text
          x={pairBox.x + pairBox.w / 2}
          y={cy - 6}
          fontSize={11}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.9}
        >
          {copy.svgPairTitle}
        </text>
        <text
          x={pairBox.x + pairBox.w / 2}
          y={cy + 9}
          fontSize={9.5}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          {copy.svgPairSub(preferred)}
        </text>
      </g>

      {/* reward model — present on the path under RLHF, greyed + bypassed under DPO */}
      <g opacity={rlhf ? 1 : 0.32}>
        <rect
          x={rewardBox.x}
          y={boxY}
          width={rewardBox.w}
          height={boxH}
          rx={6}
          fill={rlhf ? 'oklch(0.72 0.04 280 / 0.28)' : 'transparent'}
          stroke="currentColor"
          strokeOpacity={rlhf ? 0.55 : 0.35}
          strokeDasharray={rlhf ? undefined : '4 4'}
        />
        <text
          x={rewardBox.x + rewardBox.w / 2}
          y={cy - 6}
          fontSize={11}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.9}
        >
          {copy.svgReward}
        </text>
        <text
          x={rewardBox.x + rewardBox.w / 2}
          y={cy + 9}
          fontSize={9.5}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          {rlhf ? copy.svgScores : copy.svgSkipped}
        </text>
      </g>

      {/* model update */}
      <g>
        <rect
          x={updateBox.x}
          y={boxY}
          width={updateBox.w}
          height={boxH}
          rx={6}
          fill="oklch(0.72 0.15 150 / 0.18)"
          stroke="currentColor"
          strokeOpacity={0.45}
        />
        <text
          x={updateBox.x + updateBox.w / 2}
          y={cy - 6}
          fontSize={11}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.9}
        >
          {copy.svgUpdate}
        </text>
        <text
          x={updateBox.x + updateBox.w / 2}
          y={cy + 9}
          fontSize={9.5}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.6}
        >
          {rlhf ? copy.svgRlStep : copy.svgPrefLoss}
        </text>
      </g>

      {rlhf ? (
        <>
          {/* pair → reward model */}
          <line
            x1={pairR}
            y1={cy}
            x2={rewardL - 2}
            y2={cy}
            stroke="currentColor"
            strokeOpacity={0.7}
            strokeWidth={2}
            markerEnd="url(#pp-arrow)"
          />
          {/* reward model → model update */}
          <line
            x1={rewardR}
            y1={cy}
            x2={updateL - 2}
            y2={cy}
            stroke="currentColor"
            strokeOpacity={0.7}
            strokeWidth={2}
            markerEnd="url(#pp-arrow)"
          />
        </>
      ) : (
        // DPO: a single direct arrow from the pair to the model update, arcing
        // OVER the bypassed reward-model box so the "skips it" reading is literal.
        <path
          d={`M ${pairR} ${cy} C ${(pairR + updateL) / 2} ${boxY - 18}, ${(pairR + updateL) / 2} ${boxY - 18}, ${updateL - 2} ${cy}`}
          fill="none"
          stroke="oklch(0.72 0.15 150)"
          strokeOpacity={0.95}
          strokeWidth={2.5}
          markerEnd="url(#pp-arrow-direct)"
        />
      )}

      {!rlhf ? (
        <text
          x={(pairR + updateL) / 2}
          y={boxY - 22}
          fontSize={10}
          textAnchor="middle"
          fill="oklch(0.72 0.15 150)"
          fillOpacity={0.95}
        >
          {copy.svgDirect}
        </text>
      ) : null}
    </svg>
  );
}

export function PreferencePair() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();

  const [preferred, setPreferred] = React.useState<Side>('A');
  const [pA, setPA] = React.useState(P_START);
  const [updates, setUpdates] = React.useState(0);
  const [method, setMethod] = React.useState<Method>('RLHF');

  // Read `preferred` from a ref inside the functional updater so repeated
  // "apply update" clicks always nudge toward the CURRENT winner without a
  // stale-closure bug, and so the callback identity stays stable.
  const preferredRef = React.useRef(preferred);
  preferredRef.current = preferred;

  const applyUpdate = React.useCallback(() => {
    // Winner's bar UP, loser's DOWN. P(B) = 1 − P(A), so nudging P(A) by ±STEP
    // moves the two complementary bars apart by STEP each, clamped to a visible,
    // bounded band. (If A is preferred we raise P(A); if B, we lower it.)
    setPA((prev) => clamp(preferredRef.current === 'A' ? prev + STEP : prev - STEP, CLAMP_MIN, CLAMP_MAX));
    setUpdates((n) => n + 1);
  }, []);

  const reset = React.useCallback(() => {
    setPA(P_START);
    setUpdates(0);
  }, []);

  const pB = 1 - pA;
  const pAPct = pA * 100;
  const pBPct = pB * 100;

  // Winner / loser readouts are framed by who the human prefers.
  const winnerSide = preferred;
  const winPct = preferred === 'A' ? pAPct : pBPct;
  const losePct = preferred === 'A' ? pBPct : pAPct;
  const loserSide: Side = preferred === 'A' ? 'B' : 'A';

  // Bounded? Once the winner hits the clamp, further clicks can't move it.
  const atCeiling = winPct >= CLAMP_MAX * 100 - 1e-6;

  const takeaway = atCeiling
    ? copy.takeawayCeiling(winnerSide, (CLAMP_MAX * 100).toFixed(0))
    : updates === 0
      ? copy.takeawayStart(winnerSide, loserSide)
      : copy.takeawayProgress(updates, winnerSide, loserSide);

  // sr-only summary of the whole live state (the bars are DOM, not one SVG).
  const liveSummary = copy.liveSummary(preferred, pAPct.toFixed(0), pBPct.toFixed(0), updates, method);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subnote}</div>
      </div>

      {/* ---- BEAT 1: the preference loop ---- */}
      <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-2.5">
        <div className="text-[11px] text-muted-foreground">
          {copy.promptLabel}
          <span className="text-foreground/90">{copy.prompt}</span>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {(['A', 'B'] as const).map((side) => {
            const isWinner = preferred === side;
            return (
              <div
                key={side}
                className={cn(
                  'rounded-md border p-2 transition-colors',
                  isWinner ? 'border-primary/40 bg-primary/5' : 'border-border/60 bg-background',
                )}
              >
                <div className="mb-1 flex items-center justify-between">
                  <span className="font-mono text-[11px] font-medium text-foreground/90">
                    {copy.responseLabel(side)}
                  </span>
                  {isWinner ? (
                    <span className="text-[10px] uppercase tracking-wider text-primary">{copy.preferredChip}</span>
                  ) : null}
                </div>
                <p className="text-[11px] leading-relaxed text-foreground/80">
                  {side === 'A' ? copy.responseA : copy.responseB}
                </p>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-muted-foreground">{copy.humanPrefers}</span>
          <PickButton side="A" active={preferred === 'A'} onClick={() => setPreferred('A')}>
            {copy.aBetter}
          </PickButton>
          <PickButton side="B" active={preferred === 'B'} onClick={() => setPreferred('B')}>
            {copy.bBetter}
          </PickButton>
        </div>
      </div>

      {/* ---- BEAT 1: probability bars + update controls + readout ---- */}
      <div className="space-y-2">
        <ProbBar
          label={copy.probLabel('A')}
          pct={pAPct}
          bg={A_BG}
          isWinner={preferred === 'A'}
          animate={!reducedMotion}
        />
        <ProbBar
          label={copy.probLabel('B')}
          pct={pBPct}
          bg={B_BG}
          isWinner={preferred === 'B'}
          animate={!reducedMotion}
        />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={applyUpdate}
          className={cn(
            'rounded border border-border bg-muted/40 px-2.5 py-1 text-xs font-medium text-foreground/90',
            'transition-colors hover:bg-muted/70',
          )}
        >
          {copy.applyUpdate}
        </button>
        <button
          type="button"
          onClick={reset}
          className={cn(
            'rounded border border-border px-2.5 py-1 text-xs font-medium text-muted-foreground',
            'transition-colors hover:text-foreground',
          )}
        >
          {copy.reset}
        </button>
        <span className="ml-auto font-mono text-[11px] text-muted-foreground">{copy.updatesCount(updates)}</span>
      </div>

      <div className="grid grid-cols-2 gap-2" aria-live="polite">
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            {copy.chosenReadout(winnerSide)}
          </div>
          <div className="font-mono text-sm text-foreground/90">{winPct.toFixed(0)}%</div>
        </div>
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            {copy.rejectedReadout(loserSide)}
          </div>
          <div className="font-mono text-sm text-foreground/90">{losePct.toFixed(0)}%</div>
        </div>
      </div>

      <div className="text-[12px] font-medium text-foreground/90" aria-live="polite">
        {takeaway}
      </div>

      {/* Screen-reader summary of the live state (bars are DOM boxes). */}
      <p className="sr-only" aria-live="polite">
        {liveSummary}
      </p>

      {/* ---- BEAT 2: RLHF vs DPO pipeline ---- */}
      <div className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-2.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{copy.howComputed}</div>
          <SegmentedToggle
            value={method}
            onChange={setMethod}
            ariaLabel={copy.methodAria}
            options={[
              { value: 'RLHF', label: 'RLHF' },
              { value: 'DPO', label: 'DPO' },
            ]}
          />
        </div>

        <PipelineDiagram method={method} preferred={preferred} pA={pA} />

        <p className="text-[11px] leading-relaxed text-foreground/85">
          {method === 'RLHF' ? copy.rlhfExplain : copy.dpoExplain}
        </p>
      </div>

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}
