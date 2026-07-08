import * as React from 'react';

import { useUiStrings } from '../i18n/ui-react';
import type { UiStrings } from '../i18n/ui';

/**
 * A compact, non-interactive "you are here" strip shown at the top of every
 * chapter. It maps the current chapter onto the six canonical forward-pass
 * stages so the learner always knows where the concept sits in the loop that
 * turns tokens into a next-token distribution.
 *
 * Purely presentational: the only input is `chapterId`. No model/worker/WASM,
 * no animation, no state, no interactivity — so ChapterFrame's 16 callers stay
 * untouched.
 */

// A chapter's place on the forward pass. A concrete stage (or several) lights
// those chips; `all` lights every chip (the end-to-end capstone) and `meta`
// lights none (chapters that zoom out from a single step). A chapter may map to
// MORE THAN ONE stage — RMSNorm, for instance, runs both inside every block and
// as the final norm — so every visible chip is reachable as a current stage.
type ConcreteStage = 'tokenize' | 'embed' | 'block' | 'final-norm' | 'lm-head' | 'sample';
type StageSpec = ConcreteStage | readonly ConcreteStage[] | 'meta' | 'all';

// One chip per forward-pass stage, in execution order. `key` is the stable
// React key and the concrete stage this chip represents. `title` is an
// optional hover tooltip (used to flag the hybrid layer stack without
// lengthening the chip label). Labels/tooltips come from the locale dictionary
// (reusing the same strings as ChapterIndex's fixed chips / layer header).
type StageChip = { key: ConcreteStage; label: string; title?: string };

function stageChips(ui: UiStrings): readonly StageChip[] {
  return [
    { key: 'tokenize', label: ui.pipeline.chipTokenize },
    { key: 'embed', label: ui.pipeline.chipEmbedding },
    { key: 'block', label: ui.pipeline.chipLayers, title: ui.pipeline.chipLayersTitle },
    { key: 'final-norm', label: ui.pipeline.chipFinalNorm },
    { key: 'lm-head', label: ui.pipeline.chipLmHead },
    { key: 'sample', label: ui.pipeline.chipSampling },
  ];
}

// Maps each chapter id to its forward-pass stage(s). Unknown ids fall back to
// `meta` defensively (handled at the lookup site). RMSNorm maps to BOTH the
// in-block stage and the final norm so the `final-norm` chip is reachable as a
// current stage, not only lit inside the architecture capstone.
const STAGE_BY_CHAPTER: Readonly<Record<string, StageSpec>> = {
  overview: 'meta',
  tokenization: 'tokenize',
  embeddings: 'embed',
  attention: 'block',
  'multi-head-gqa': 'block',
  rope: 'block',
  rmsnorm: ['block', 'final-norm'],
  mlp: 'block',
  'full-block': 'block',
  'kv-cache': 'block',
  'lm-head': 'lm-head',
  sampling: 'sample',
  training: 'meta',
  scaling: 'meta',
  'post-training': 'meta',
  architecture: 'all',
};

// The concrete chips a spec lights (empty for `meta`/`all`, which branch separately).
function litStages(spec: StageSpec): readonly ConcreteStage[] {
  if (spec === 'meta' || spec === 'all') return [];
  return typeof spec === 'string' ? [spec] : spec;
}

// aria-label sentence, built from the locale dictionary (stage phrases,
// pipeline summary, and the sentence templates all localize together).
function describe(ui: UiStrings, spec: StageSpec): string {
  if (spec === 'all') {
    return ui.pipeline.describeAll(ui.pipeline.summary);
  }
  if (spec === 'meta') {
    return ui.pipeline.describeMeta(ui.pipeline.summary);
  }
  const keys = litStages(spec);
  const phrase = keys.map((k) => ui.pipeline.stagePhrases[k] ?? k).join(ui.pipeline.phraseJoin);
  return ui.pipeline.describeStages(phrase, keys.length > 1, ui.pipeline.summary);
}

export function PipelineCue({ chapterId }: { chapterId: string }) {
  const ui = useUiStrings();
  const spec: StageSpec = STAGE_BY_CHAPTER[chapterId] ?? 'meta';
  const isMeta = spec === 'meta';
  const isAll = spec === 'all';
  const litKeys = new Set<ConcreteStage>(litStages(spec));

  return (
    <div role="img" aria-label={describe(ui, spec)} className={isMeta ? 'opacity-50' : undefined}>
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{ui.pipeline.eyebrow}</span>
        {stageChips(ui).map((chip, i) => {
          const lit = isAll || litKeys.has(chip.key);
          return (
            <React.Fragment key={chip.key}>
              {i > 0 && (
                <span aria-hidden="true" className="text-muted-foreground">
                  →
                </span>
              )}
              <span
                title={chip.title}
                className={[
                  'rounded border px-1.5 py-0.5 text-[10px] sm:text-[11px]',
                  lit
                    ? 'border-primary bg-primary/10 text-foreground font-medium'
                    : 'border-border bg-background text-muted-foreground',
                ].join(' ')}
              >
                {chip.label}
              </span>
            </React.Fragment>
          );
        })}
        {isMeta && <span className="text-[10px] text-muted-foreground">{ui.pipeline.wholePipeline}</span>}
      </div>
    </div>
  );
}
