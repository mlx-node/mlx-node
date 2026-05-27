import * as React from "react";

/**
 * Three side-by-side mini panels demonstrating common sampling failure modes
 * on the same prompt. Pre-recorded, illustrative continuations — no live
 * model run — so the teaching contrast is stable visit-to-visit.
 */

const PROMPT = "Once upon a time, in a forest far away";

type Panel = {
  label: string;
  config: string;
  continuation: string;
  takeaway: string;
};

const PANELS: Panel[] = [
  {
    label: "Greedy",
    config: "T = 0, top-p = 1.0",
    continuation:
      " there lived a small forest there lived a small forest",
    takeaway:
      "Repetition trap. The model finds a high-confidence phrase and loops back into it.",
  },
  {
    label: "Too hot",
    config: "T = 2.0",
    continuation:
      " a frgg moo whirr the of bicycle banana very",
    takeaway:
      "Gibberish. The distribution is so flat the sampler picks rare tokens uniformly.",
  },
  {
    label: "Sweet spot",
    config: "T = 0.7, top-p = 0.9",
    continuation:
      " a small village where everyone knew each other and shared their stories",
    takeaway:
      "Coherent yet varied. Top-p trims the absurd tail, temperature keeps it from collapsing.",
  },
];

export function SamplingFailureModes() {
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Sampling failure modes · same prompt, three regimes
        </div>
        <div className="text-[11px] text-muted-foreground">
          Pre-recorded continuations · 10 tokens each
        </div>
      </div>

      <div className="rounded-md bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
        Prompt: <span className="text-foreground/90">{JSON.stringify(PROMPT)}</span>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {PANELS.map((p, i) => (
          <div
            key={`panel-${i}`}
            className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3"
          >
            <div className="flex items-baseline justify-between gap-2">
              <div className="font-mono text-[12px] font-semibold text-foreground">
                {p.label}
              </div>
              <div className="font-mono text-[10px] text-muted-foreground">
                {p.config}
              </div>
            </div>
            <div className="rounded-md border border-border/50 bg-background px-2 py-2 font-mono text-[12px] leading-relaxed">
              <span className="text-muted-foreground">{PROMPT}</span>
              <span className="text-primary">{p.continuation}</span>
            </div>
            <p className="text-[11px] text-muted-foreground">{p.takeaway}</p>
          </div>
        ))}
      </div>

      <p className="text-[12px] text-foreground/85">
        Production LLM serving typically lands somewhere around{" "}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
          T = 0.7-1.0
        </code>{" "}
        with{" "}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
          top-p = 0.9
        </code>{" "}
        (or a moderate top-K). The two knobs do different jobs: temperature
        reshapes the whole distribution, top-p trims the long tail. Together
        they avoid the greedy loop and the hot-gibberish failure modes you
        see above.
      </p>
    </div>
  );
}
