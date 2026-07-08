/**
 * J-lens QUALITATIVE PROBE (Task T1.4) — J-vs-logit argmax grids + a cheap
 * workspace-onset detector (residual-stream tail index by boundary).
 *
 * PART A — argmax grids. For 4 hand-picked prompts (a sport verbal-report, an
 * arithmetic order-of-ops, a two-hop spider prompt, and a plain continuation)
 * we run BOTH lenses over boundaries 1..=24 and record the top-1 token at every
 * (layer, position) cell. The interesting slice — the readout column, printed
 * layer-by-layer J vs logit — shows WHERE the J-lens surfaces an interpretable
 * intermediate that the logit lens does not. For each prompt we also pin the
 * known intermediate concept and report its J-vs-logit min-over-layers rank, so
 * the divergence is a hard number, not a vibe.
 *
 * PART B — workspace-onset detector. The paper's cheapest onset signal
 * (Fig-28) is the EXCESS KURTOSIS of the residual stream by layer. The
 * read-only inspector NAPI exposes per-layer summary stats (mean/std/absMax/
 * l2) but NOT the raw residual activations, and this task makes NO native
 * changes — so we report the closest available proxy: the residual TAIL INDEX
 * absMax/std of `post_mlp_residual` by boundary (max standardized activation).
 * It is monotone-aligned with excess kurtosis (both spike exactly when a few
 * massive activations appear) and answers the same question — WHERE does the
 * residual stream develop heavy-tailed / outlier structure past the early band.
 * This is a PROXY, explicitly labeled; it is NOT the literal 4th moment.
 *
 * Run with: oxnode packages/browser/scripts/jlens/probe.mts
 *   (NOT tsx/ts-node; needs env PATH="/opt/homebrew/bin:$PATH")
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const PACK_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-pilot.safetensors';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';

const REQUEST_LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const N_HEADLINE = 23; // layers 1..23 = fitted-J domain
const RANK_CENSOR = 999;

type Probe = { label: string; prompt: string; readout: 'preTarget' | 'lastNewline'; concepts: string[] };
const PROBES: Probe[] = [
  {
    label: 'sport-verbal-report',
    prompt: 'Fact: The number of players per side in the sport invented in Springfield, Massachusetts is ',
    readout: 'preTarget',
    concepts: ['basketball'],
  },
  { label: 'order-of-ops-calc', prompt: 'calc: (4+17)*2+7=', readout: 'preTarget', concepts: ['21', '42'] },
  {
    label: 'spider-two-hop',
    prompt: 'Fact: The number of legs on the animal that spins webs is ',
    readout: 'preTarget',
    concepts: ['spider'],
  },
  {
    label: 'plain-continuation',
    prompt: 'The quick brown fox jumps over the lazy ',
    readout: 'preTarget',
    concepts: ['dog'],
  },
];

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  if (loaded !== 23) throw new Error(`expected 23 Jacobians, got ${loaded}`);
  console.log(`Model + pilot pack (${loaded} Jacobians) loaded.\n`);

  const singleTokIds = async (base: string): Promise<number[]> => {
    const ids = new Set<number>();
    for (const cand of [` ${base}`, base]) {
      const e: number[] = await model.encodeTokens(cand);
      if (e.length === 1) ids.add(e[0]);
    }
    return [...ids];
  };

  const gridsOut: any[] = [];

  for (const probe of PROBES) {
    const ids: number[] = await model.encodeTokens(probe.prompt);
    const texts: string[] = await model.decodeTokens(ids);
    const P = ids.length;
    let readoutPos = P - 1;
    if (probe.readout === 'lastNewline') {
      for (let i = 0; i < P; i++) if (texts[i].includes('\n')) readoutPos = i;
    }

    // concept surface ids (for the crisp min-rank number).
    const conceptPins: { concept: string; ids: number[] }[] = [];
    for (const c of probe.concepts) conceptPins.push({ concept: c, ids: await singleTokIds(c) });
    const pinned = [...new Set(conceptPins.flatMap((c) => c.ids))].slice(0, 8);

    // full argmax grids (both lenses) + pinned ranks in one call each.
    const runOne = async (useJacobian: boolean) => {
      const ro = await model.lensReadout(ids, { layers: REQUEST_LAYERS, topK: 1, pinnedIds: pinned, useJacobian });
      // layer-major then position-minor
      const argmax: string[][] = REQUEST_LAYERS.map((_, li) =>
        Array.from({ length: P }, (_, p) => ro.cells[li * P + p].topKTexts[0] ?? ''),
      );
      return { ro, argmax };
    };
    const j = await runOne(true);
    const l = await runOne(false);

    console.log(`\n================ ${probe.label} ================`);
    console.log(`prompt=${JSON.stringify(probe.prompt)}`);
    console.log(`P=${P} readoutPos=${readoutPos} tokenThere=${JSON.stringify(texts[readoutPos])}`);

    // readout-column: J vs logit argmax by layer.
    console.log(`  layer |     J-lens argmax     |    logit argmax`);
    for (let li = 0; li < REQUEST_LAYERS.length; li++) {
      const ly = REQUEST_LAYERS[li];
      const jt = j.argmax[li][readoutPos];
      const lt = l.argmax[li][readoutPos];
      const mark = jt !== lt ? '  <-' : '';
      console.log(
        `  ${String(ly).padStart(2)}    | ${JSON.stringify(jt).padEnd(22)}| ${JSON.stringify(lt).padEnd(18)}${mark}`,
      );
    }

    // concept min-rank (J vs logit) at the readout position.
    const minRank = (ro: any, cids: number[]) => {
      let best = RANK_CENSOR;
      let bestLayer = -1;
      for (let bi = 0; bi < pinned.length; bi++) {
        if (!cids.includes(pinned[bi])) continue;
        const ranks = ro.pinned[bi].ranks as Int32Array;
        for (let li = 0; li < N_HEADLINE; li++) {
          const r = ranks[li * P + readoutPos];
          if (r < best) {
            best = r;
            bestLayer = REQUEST_LAYERS[li];
          }
        }
      }
      return { best, bestLayer };
    };
    const conceptReport: any[] = [];
    for (const c of conceptPins) {
      if (c.ids.length === 0) {
        // multi-token concept (e.g. 2-digit numbers under digit-level BPE): can
        // NOT be pinned as a single vocab id — the argmax grid above is the
        // qualitative evidence for it, not a pinned rank. Report honestly.
        console.log(
          `  concept "${c.concept}": NOT single-token pinnable (multi-token under this tokenizer) — see argmax grid, not a rank`,
        );
        conceptReport.push({ concept: c.concept, ids: [], pinnable: false });
        continue;
      }
      const jm = minRank(j.ro, c.ids);
      const lm = minRank(l.ro, c.ids);
      const verdict = jm.best < lm.best ? 'J surfaces it earlier/better' : jm.best === lm.best ? 'tie' : 'logit better';
      console.log(
        `  concept "${c.concept}" (ids [${c.ids.join(',')}]): ` +
          `J min-rank=${jm.best}${jm.bestLayer > 0 ? `@L${jm.bestLayer}` : ''}  logit min-rank=${lm.best}  -> ${verdict}`,
      );
      conceptReport.push({
        concept: c.concept,
        ids: c.ids,
        pinnable: true,
        jMinRank: jm.best,
        jLayer: jm.bestLayer,
        logitMinRank: lm.best,
      });
    }

    gridsOut.push({
      label: probe.label,
      prompt: probe.prompt,
      tokens: texts,
      readoutPos,
      layers: REQUEST_LAYERS,
      jArgmax: j.argmax,
      logitArgmax: l.argmax,
      conceptReport,
    });
  }

  // ---- PART B: residual tail-index (kurtosis proxy) by boundary ----
  console.log(`\n================ residual tail-index by boundary (excess-kurtosis PROXY) ================`);
  console.log(`(absMax/std of post_mlp_residual; PROXY for the paper's Fig-28 excess-kurtosis onset detector.`);
  console.log(
    ` raw residual activations are NOT exposed by the read-only NAPI and this task makes no native changes.)`,
  );
  // boundary ℓ (1..24) == post_mlp_residual at decoder layerIdx ℓ-1.
  const nB = 24;
  const tailSum = new Array(nB).fill(0);
  const stdSum = new Array(nB).fill(0);
  const absMaxSum = new Array(nB).fill(0);
  let nP = 0;
  for (const probe of PROBES) {
    const hs = await model.runForInspector(probe.prompt, {
      hiddenStates: { points: ['post_mlp_residual'] },
      maxNewTokens: 1,
      applyChatTemplate: false,
    });
    const step0 = hs.hiddenStates[0];
    for (const lay of step0.layers) {
      const s = lay.stats.find((x: any) => x.point === 'post_mlp_residual');
      if (!s) continue;
      const b = lay.layerIdx; // boundary index ℓ-1 -> boundary ℓ = layerIdx+1
      tailSum[b] += s.absMax / s.std;
      stdSum[b] += s.std;
      absMaxSum[b] += s.absMax;
    }
    nP++;
  }
  const tail = tailSum.map((v) => v / nP);
  const stdMean = stdSum.map((v) => v / nP);
  const absMaxMean = absMaxSum.map((v) => v / nP);
  const earlyBandMean = tail.slice(0, 5).reduce((a, b) => a + b, 0) / 5; // boundaries 1..5
  console.log(`  boundary | tail(absMax/std) | std     | absMax`);
  for (let b = 0; b < nB; b++) {
    const rise = tail[b] > earlyBandMean * 1.25 ? '  ^rise' : '';
    console.log(
      `  ${String(b + 1).padStart(2)}       | ${tail[b].toFixed(2).padStart(6)}           | ${stdMean[b].toFixed(4)} | ${absMaxMean[b].toFixed(3)}${rise}`,
    );
  }
  console.log(`  early-band (ℓ1..5) mean tail-index = ${earlyBandMean.toFixed(2)}; boundaries >1.25× flagged ^rise.`);

  // tiny standalone SVG of the tail-index curve.
  const svgW = 640,
    svgH = 220,
    padL = 44,
    padB = 28,
    padT = 14,
    padR = 10;
  const maxT = Math.max(...tail) * 1.1;
  const xOf = (b: number) => padL + (b / (nB - 1)) * (svgW - padL - padR);
  const yOf = (t: number) => svgH - padB - (t / maxT) * (svgH - padB - padT);
  const pts = tail.map((t, b) => `${xOf(b).toFixed(1)},${yOf(t).toFixed(1)}`).join(' ');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">
<rect width="${svgW}" height="${svgH}" fill="#0b0e14"/>
<text x="${svgW / 2}" y="12" fill="#c9d1d9" font-family="sans-serif" font-size="11" text-anchor="middle">residual tail-index absMax/std by boundary (excess-kurtosis proxy) — mean over 4 probes</text>
<line x1="${padL}" y1="${svgH - padB}" x2="${svgW - padR}" y2="${svgH - padB}" stroke="#30363d"/>
<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${svgH - padB}" stroke="#30363d"/>
<line x1="${padL}" y1="${yOf(earlyBandMean).toFixed(1)}" x2="${svgW - padR}" y2="${yOf(earlyBandMean).toFixed(1)}" stroke="#f0883e" stroke-dasharray="4 3" opacity="0.7"/>
<text x="${svgW - padR}" y="${(yOf(earlyBandMean) - 3).toFixed(1)}" fill="#f0883e" font-family="sans-serif" font-size="9" text-anchor="end">early-band mean ${earlyBandMean.toFixed(1)}</text>
<polyline points="${pts}" fill="none" stroke="#58a6ff" stroke-width="1.8"/>
${tail.map((t, b) => `<circle cx="${xOf(b).toFixed(1)}" cy="${yOf(t).toFixed(1)}" r="2" fill="#58a6ff"/>`).join('')}
${[1, 6, 12, 18, 24].map((ly) => `<text x="${xOf(ly - 1).toFixed(1)}" y="${svgH - 10}" fill="#8b949e" font-family="sans-serif" font-size="9" text-anchor="middle">${ly}</text>`).join('')}
</svg>`;

  mkdirSync(OUT_DIR, { recursive: true });
  writeFileSync(
    join(OUT_DIR, 'probe-grids.json'),
    JSON.stringify(
      {
        model: MODEL_PATH,
        pack: PACK_PATH,
        generatedAt: new Date().toISOString(),
        grids: gridsOut,
        tailIndex: {
          boundaries: REQUEST_LAYERS,
          tail,
          stdMean,
          absMaxMean,
          earlyBandMean,
          note: 'PROXY for excess kurtosis (raw residuals not exposed by read-only NAPI)',
        },
      },
      null,
      2,
    ),
  );
  writeFileSync(join(OUT_DIR, 'probe-tail-index.svg'), svg);
  console.log(`\nWrote ${join(OUT_DIR, 'probe-grids.json')} and probe-tail-index.svg`);
  console.log('PROBE COMPLETE');
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
