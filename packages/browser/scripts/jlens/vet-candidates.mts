/**
 * Vet all candidate prompts through logit + fitted-Jacobian lens.
 * Run: env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors \
 *   oxnode scripts/jlens/vet-candidates.mts
 */
const MODEL_PATH = '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16';
const OUT_DIR = '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens';
import { join } from 'node:path';
import { writeFileSync, readFileSync } from 'node:fs';
const PACK_ENV = process.env.JLENS_PACK ?? 'lens-pack-v1.safetensors';
const PACK_PATH = PACK_ENV.includes('/') ? PACK_ENV : join(OUT_DIR, PACK_ENV);

const CANDIDATES = JSON.parse(readFileSync(process.env.CAND_FILE!, 'utf8')) as {
  name: string;
  candidates: { prompt: string; expectedIntermediateConcept: string; pinTokens: string[]; rationale: string }[];
}[];

const LAYERS = Array.from({ length: 24 }, (_, i) => i + 1); // 1..24
const TOP_K = 8;

function norm(s: string): string {
  return (s ?? '').trim().toLowerCase();
}

async function main() {
  const { Qwen35Model } = await import('@mlx-node/lm');
  console.log(`Loading model from ${MODEL_PATH} ...`);
  const model = (await Qwen35Model.load(MODEL_PATH)) as any;
  console.log(`Loading lens pack from ${PACK_PATH} ...`);
  const loaded = await model.loadLensPackFromFile(PACK_PATH);
  console.log(`pack Jacobians = ${loaded}\n`);

  const results: any[] = [];

  for (const group of CANDIDATES) {
    for (const cand of group.candidates) {
      const promptIds: number[] = await model.encodeTokens(cand.prompt);
      const promptTexts: string[] = await model.decodeTokens(promptIds);
      const promptNorm = new Set(promptTexts.map(norm));

      const rec: any = {
        phenomenon: group.name,
        prompt: cand.prompt,
        expected: cand.expectedIntermediateConcept,
        pinTokens: cand.pinTokens,
        promptTexts,
        lenses: {},
      };

      for (const useJacobian of [false, true]) {
        const ro = (await model.lensReadout(promptIds, {
          layers: LAYERS,
          topK: TOP_K,
          pinnedIds: [],
          useJacobian,
        })) as any;
        const P = ro.promptLen;
        const lastPos = P - 1;
        // final-layer argmax at last pos = model's actual greedy next-token
        const finalCell = ro.cells[(ro.layers.length - 1) * P + lastPos];
        const greedy = finalCell.topKTexts[0];

        // per-layer top-K at last position
        const perLayer: { layer: number; toks: string[] }[] = [];
        for (let li = 0; li < ro.layers.length; li++) {
          const cell = ro.cells[li * P + lastPos];
          perLayer.push({ layer: ro.layers[li], toks: cell.topKTexts.slice() });
        }

        // For each pin token: find layers where a top-K text matches (normalized equal OR contains)
        const pinHits: Record<string, { layer: number; rank: number }[]> = {};
        for (const pin of cand.pinTokens) {
          const pn = norm(pin);
          const hits: { layer: number; rank: number }[] = [];
          for (const row of perLayer) {
            for (let r = 0; r < row.toks.length; r++) {
              const tn = norm(row.toks[r]);
              if (tn === pn || (pn.length >= 3 && tn === pn)) {
                hits.push({ layer: row.layer, rank: r });
                break;
              }
            }
          }
          pinHits[pin] = hits;
        }

        rec.lenses[useJacobian ? 'jacobian' : 'logit'] = {
          greedy,
          promptLen: P,
          perLayer,
          pinHits,
        };
      }

      // absence checks: is expected concept in prompt? in greedy output?
      const expNorm = norm(cand.expected);
      rec.expectedInPrompt = [...promptNorm].some((t) => t === expNorm);
      results.push(rec);

      // console summary
      console.log(`\n########## [${group.name}] ${JSON.stringify(cand.prompt)}`);
      console.log(`  expected intermediate: "${cand.expected}"  pins: ${cand.pinTokens.join(' | ')}`);
      console.log(`  prompt tokens: ${promptTexts.map((t) => JSON.stringify(t)).join(' ')}`);
      for (const lens of ['logit', 'jacobian'] as const) {
        const L = rec.lenses[lens];
        console.log(`  --- ${lens.toUpperCase()} lens (greedy next = ${JSON.stringify(L.greedy)}) ---`);
        for (const pin of cand.pinTokens) {
          const h = L.pinHits[pin];
          if (h.length) {
            const best = h.reduce((a, b) => (b.rank < a.rank ? b : a));
            console.log(
              `      pin ${JSON.stringify(pin)}: hits L[${h.map((x) => `${x.layer}#${x.rank}`).join(',')}]  best=L${best.layer}#${best.rank}`,
            );
          } else {
            console.log(`      pin ${JSON.stringify(pin)}: (none)`);
          }
        }
      }
    }
  }

  const outPath = join(OUT_DIR, 'vet-candidates-results.json');
  writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`\n\nWROTE ${outPath}`);
  process.exit(0);
}
main().catch((e) => {
  console.error('FAILED:', e);
  process.exit(1);
});
