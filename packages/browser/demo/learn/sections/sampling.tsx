// Sub-chapter 11.1 — Multi-token prediction & self-speculative decoding.
//
// A "go deeper" deep-dive hung off chapter 11 (Sampling). Pure, model-free
// reading prose — SSR-safe, so the prerenderer renders it straight to crawlable
// static HTML (see scripts/prerender.entry.tsx). Wrapped in <Prose> by the
// section route / prerenderer, so it returns just the article content.
//
// Every architectural claim is grounded in the real Qwen3.5 MTP head:
//   - config.json: mtp_num_hidden_layers 1, mtp_use_dedicated_embeddings false,
//     tie_word_embeddings true — one shared sequential module, depth 1.
//   - The module (verified against the native engine + vLLM/SGLang qwen3_next_mtp):
//     RMSNorm(prev_hidden) + RMSNorm(emb) -> concat -> fc(2H->H, no bias) ->
//     one FULL-attention decoder layer -> RMSNorm -> the main model's tied LM head.
//   - Lineage numbers are primary-sourced: Meta arXiv 2404.19737 (n=4 parallel
//     heads, up to 3x, +12% HumanEval / +17% MBPP at 13B); DeepSeek-V3 arXiv
//     2412.19437 (D=1 sequential, keeps the causal chain, loss weight lambda
//     0.3 -> 0.1); Qwen blog (Qwen3.5 = Qwen3-Next, MTP boosts pretraining
//     efficiency AND inference speed).
//   - Honesty: this browser's bf16 checkpoint declares mtp_num_hidden_layers 1
//     but ships zero mtp.* tensors, and the WebGPU decode path is plain
//     autoregressive; the native engine runs the full draft/verify loop
//     (measured ~1.06-1.46x there). HF transformers also discards MTP weights.

import * as React from 'react';

import { Prose } from '../Prose';
import { ChapterLink } from '../scaffolding/ChapterLink';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { ConstrainedDecodeMask } from '../widgets/ConstrainedDecodeMask';
import { MtpLineageDiagram } from '../widgets/MtpLineageDiagram';
import { MtpModuleDiagram } from '../widgets/MtpModuleDiagram';
import { SelfConsistencyVoting } from '../widgets/SelfConsistencyVoting';
import { SpecDecodeFlamechart } from '../widgets/SpecDecodeFlamechart';
import { SpecDecodeVariantsDiagram } from '../widgets/SpecDecodeVariantsDiagram';
import { SpeculativeDecodeDiagram } from '../widgets/SpeculativeDecodeDiagram';
import { SpeculativeVerifyLive } from '../widgets/SpeculativeVerifyLive';

/** Sub-chapter 11.1 — multi-token prediction and the speculative decoding it powers. */
export function SamplingMtpSection() {
  return (
    <Prose>
      <h1>Multi-token prediction</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 11, Sampling — how a model gets <em>more than one token</em> out of a single forward pass.
      </p>
      <p>
        Sampling, from the last chapter, turns one row of logits into one token. To produce the next token you append it
        to the prompt and run <em>the entire model again</em> — all 24 layers, every weight. A 200-token reply is 200
        full forward passes, strictly one after another, because each token depends on the one before it. That is the{' '}
        <strong>autoregressive tax</strong>, and it is the single biggest reason generation feels slow.
      </p>
      <p>
        What makes it hurt is <em>what</em> each pass spends its time on. Producing one token touches every weight in the
        model exactly once, so a decode step is <strong>memory-bound</strong>: the GPU spends most of the step just
        reading the billion-ish parameters out of memory, and the actual math for that single token is a rounding error
        on top. The chip is mostly idle, waiting on memory. So here is the tempting question: if we are going to pay to
        stream all those weights in anyway, can we get <em>more than one token</em> for that price?
      </p>

      <h2>The bet: draft, then verify</h2>
      <p>
        <strong>Speculative decoding</strong> is the trick that says yes. The idea is to <em>guess</em> the next few
        tokens cheaply, then have the real model <em>check all the guesses at once</em> in a single forward pass.
        Checking <code>k</code> guesses costs about the same as producing one token normally — it is one pass over the
        weights — so every guess that survives is a token you got for free. Guess well and you emit several tokens per
        pass; guess badly and you have wasted a little cheap drafting, nothing more.
      </p>
      <p>
        Classically you need a <em>second, smaller</em> &ldquo;draft model&rdquo; to do the guessing — an extra model to
        train, load, and keep in sync. <strong>Multi-token prediction</strong> (MTP) removes that cost: the model drafts
        for <em>itself</em>. Qwen3.5 ships one small extra module — the MTP head — whose only job is to propose the{' '}
        <em>next</em> token while the main model is still finishing the current one.
      </p>

      <h2>The MTP head Qwen3.5 carries</h2>
      <p>
        It is deliberately tiny, because it borrows almost everything from the main model. It takes <em>two</em> inputs:
        the main model&apos;s <strong>last hidden state</strong> for the current position, and the <strong>embedding of
        the token that was just emitted</strong> — and crucially that embedding comes from the <em>same</em> embedding
        matrix the main model uses (<code>mtp_use_dedicated_embeddings: false</code>). It normalizes both, fuses them with
        a single projection, runs them through <em>one</em> transformer layer, and reads out a token through the{' '}
        <em>same tied LM head</em> the main model uses. That is the whole head:
      </p>

      <MtpModuleDiagram />

      <p>
        Two details matter for later. First, the head&apos;s one layer is a <strong>full-attention</strong> layer — not
        one of the 18 GatedDeltaNet layers — and it keeps its own little attention cache; it never touches the main
        model&apos;s recurrent state. Second, because the embedding and the output head are <em>shared</em>, the only new
        weights are three RMSNorms, one projection, and that single layer. In the config this is one line —{' '}
        <code>mtp_num_hidden_layers: 1</code> — a single module that predicts <em>one</em> token beyond the next.
      </p>

      <h2>The loop: draft <MathDisplay inline latex={String.raw`D`} />, verify once, accept what matches</h2>
      <p>
        Now put the head to work. One decode step becomes three moves. <strong>Draft:</strong> the cheap MTP head proposes{' '}
        <code>D</code> candidate tokens, one after another (Qwen3.5&apos;s default depth is <code>D = 1</code>, though an
        engine can raise it). <strong>Verify:</strong> the full model takes the window{' '}
        <code>[last committed token, d₁, …, d_D]</code> and runs it through in <em>one</em> batched forward pass, getting
        its own opinion at every position at once. <strong>Accept:</strong> walk the draft left to right and keep each
        guess that matches what the full model would have picked; at the first mismatch, emit a correction instead and
        stop — at temperature 0 that correction is simply the full model&apos;s top token, while with sampling it is drawn
        from an adjusted distribution built from both models&apos; probabilities, not literally the target&apos;s own pick.
        Step through it:
      </p>

      <SpeculativeDecodeDiagram />

      <p>
        Count the payoff. If <code>K</code> of the <code>D</code> drafts are accepted, the step emits{' '}
        <code>K + 1</code> tokens — the <code>K</code> good guesses plus one token the verify pass produced for free (the
        correction at the first miss, or a bonus token past the end if <em>all</em> drafts were right). So one pass over
        the weights yields anywhere from 1 token (everything rejected — the same single full-model pass as plain
        decoding, plus the small wasted cost of the draft) up to <code>D + 1</code>.
      </p>

      <h2>Why it is exactly lossless</h2>
      <p>
        Here is the part that makes speculative decoding honest rather than a quality shortcut: at temperature 0, a
        draft is kept <em>only</em> when it matches the full model&apos;s own top pick at that position. With sampling,
        the accept test is different but just as strict — it compares target and draft probabilities for that exact
        drafted token and accepts with probability{' '}
        <MathDisplay inline latex={String.raw`\min(1, p_{\text{target}}/p_{\text{draft}})`} />, resampling from a
        corrective residual distribution on reject. Either way the verify pass is the real model&apos;s real
        distribution; the MTP head never gets the last word. With the proper acceptance rule —
        the speculative sampling/decoding algorithm from Leviathan et al. and, independently, Chen et al., both 2023 —
        the tokens you emit follow the{' '}
        <strong>exact same distribution as plain one-at-a-time decoding</strong> — the same probabilities at every step
        and the same temperature behaviour, just produced in fewer passes (at temperature 0 that means the identical
        text; with sampling, the identical odds). The MTP head never changes the answer — at worst it costs a small
        drafting tax, at best it saves several passes.
      </p>
      <p className="text-muted-foreground">
        That is also why a wrong guess is cheap: a rejected draft costs only the little bit of MTP compute that produced
        it. The expensive verify pass was going to happen regardless — it is the same forward pass plain decoding would
        have run to produce that one token. Speculative decoding is, at worst, plain decoding with a small drafting tax;
        at best, a multiple of the speed.
      </p>

      <h2>Where the idea came from</h2>
      <p>
        MTP did not arrive fully formed. Three steps got us here, and the design Qwen3.5 ships is the third:
      </p>

      <MtpLineageDiagram />

      <p>
        Meta&apos;s 2024 version bolted <em>four parallel heads</em> onto a shared trunk, each predicting a different
        future token at the same time — great as a training signal, but the heads are independent, so they cannot
        condition on each other. DeepSeek-V3 changed the shape to a <em>single sequential module</em> that predicts one
        extra token while <em>keeping the causal chain</em> — the draft of token <em>t+2</em> gets to see the chosen{' '}
        <em>t+1</em>. Qwen3.5 (built on the Qwen3-Next architecture) adopts exactly that sequential, weight-sharing
        design: one module, depth one.
      </p>

      <h2>It earns its keep twice</h2>
      <p>
        The same head pays off at two completely different times. During <strong>training</strong>, asking each position
        to predict the next <em>two</em> tokens instead of one is a denser learning signal — more to learn from every
        token of data. Meta reported their multi-token models solving <strong>12% more</strong> HumanEval and{' '}
        <strong>17% more</strong> MBPP problems at 13B; DeepSeek-V3 folds an MTP loss into pretraining with a weight that
        starts at <MathDisplay inline latex={String.raw`\lambda = 0.3`} /> and drops to{' '}
        <MathDisplay inline latex={String.raw`\lambda = 0.1`} /> for the final stretch of tokens:
      </p>
      <MathDisplay latex={String.raw`\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^{k}`} />
      <p>
        Qwen describes its own MTP as boosting <em>both</em> pretraining efficiency and inference speed. During{' '}
        <strong>inference</strong>, the very same weights become the free self-drafter from the loop above. One module,
        two jobs: a better-trained model, and a faster one.
      </p>

      <h2>Where the time goes</h2>
      <p>
        Earlier we said a decode step is memory-bound &mdash; the chip mostly idle, waiting for weights. Speculative
        decoding only makes sense once you <em>see</em> that. Below is the story the way a GPU profiler would draw it:
        two timelines on one clock, plain decoding on top, draft &rarr; verify &rarr; accept underneath. Each pass is
        &asymp;20&nbsp;ms wide because that is roughly what this tab&apos;s own plain decode measures (~50 tok/s on an
        M-series Mac) &mdash; and the verify pass is <em>just as wide</em> as a plain one, because either way it is one
        pass over the weights. Raise the draft depth, make the text harder to guess, and watch what happens to the
        trace:
      </p>

      <SpecDecodeFlamechart />

      <p>
        Two things to notice. First, <code>D</code> has diminishing returns: draft number four only counts if drafts
        one through three all survived, so each extra guess is worth less than the one before it. Second,
        predictability gates everything &mdash; on hard-to-guess text the speculative lane collapses back to plain
        decoding plus a &asymp;1&nbsp;ms drafting tax, no matter how deep you draft. And look at what the red slots do
        to the timeline: nothing. Rejected positions never widen a pass &mdash; the waste rides a memory sweep you were
        paying for anyway. That is also why the trick pays at batch 1 and gets switched off on saturated servers: the
        free ride exists only while the compute row is empty.
      </p>

      <h2>A zoo of drafters</h2>
      <p>
        Everything above used the MTP head as the source of the cheap draft. But the{' '}
        <strong>draft → verify → accept</strong> skeleton doesn&apos;t care <em>where</em> the guess comes from — only
        that it&apos;s cheap, and that the verify pass keeps the lossless guarantee. Swap out the drafter and you get a
        whole family of <strong>speculative decoding</strong>, all sharing the exact same verify pass you just stepped
        through.
      </p>
      <p>
        There&apos;s a <strong>draft-target</strong> setup (a second, smaller model writes the draft — easiest to bolt
        on, but you now run two models); <strong>Medusa</strong> (grow a few extra prediction heads on the model itself,
        no separate model); <strong>EAGLE</strong> (a purpose-built tiny model fed the target&apos;s hidden states — the
        go-to for general chat); and <strong>n-gram / lookahead</strong> decoding (no draft model at all — just look up
        what usually follows, which wins big on code and predictable syntax). The MTP head you just met is the one
        Qwen3.5 ships. Pick a drafter and compare what changes — and what doesn&apos;t:
      </p>

      <SpecDecodeVariantsDiagram />

      <p>
        Three rules hold across the whole zoo. They are <strong>all lossless</strong> — at temperature 0 verify keeps a
        draft only when it matches the real model&apos;s own top pick, and with sampling it runs the same
        probability-ratio accept/resample test from above, so every variant changes <em>speed, never the answer</em>.
        They all help
        most at <strong>low batch sizes</strong>, where the GPU has spare compute to run the verify pass for free; at
        high batch the chip is already saturated (the{' '}
        <a href="/chapters/kv-cache/batching">batching</a> sub-chapter shows why), so servers turn speculation{' '}
        <em>off</em>. And higher <strong>temperature</strong> makes the next token harder to predict, which lowers the
        acceptance rate. Speculation buys speed when you&apos;re idle and the text is predictable, and quietly steps
        aside when it isn&apos;t.
      </p>

      <h2>Watch the real model verify — in one pass</h2>
      <p>
        Everything above was illustrative — canned tokens, scripted verdicts. This one is <strong>live</strong>: the
        actual Qwen3.5-0.8B you can load into this tab verifies a cheap draft in a <em>single real forward pass</em>,
        and every verdict below comes from the model&apos;s own logits. The draft comes from{' '}
        <strong>prompt-lookup</strong> — the n-gram drafter from the zoo above, really implemented here: the last few
        tokens of the prefix are looked up earlier in the prefix, and whatever followed there becomes the guess — or
        from your own typed continuation. It does <em>not</em> come from Qwen&apos;s MTP head; this checkpoint ships
        zero <code>mtp.*</code> tensors, as the next section explains. One more honesty note: the prefix is fed to the
        model as raw text, with no chat template — a raw language-model continuation, not a chat turn.
      </p>

      <SpeculativeVerifyLive />

      <p>
        Three things to notice. The verify is greedy, temperature 0 — a draft token is accepted <em>exactly</em> when
        it is the model&apos;s own argmax at that position, nothing softer. On a reject, the model&apos;s own pick (the
        green correction chip) is what continues the text — that is why speculative decoding is lossless: the output is
        what plain decoding would have produced anyway. And the whole check — every position, every top-K distribution
        you can expand above — cost <strong>one</strong> forward pass over the weights. That single pass is the entire
        speed story this sub-chapter has been telling. (This widget is a one-off verify probe wired up for this demo —
        the browser&apos;s actual chat decode loop stays plain autoregressive, as the next section explains.)
      </p>

      <h2>Does your in-browser Qwen use it?</h2>
      <p>
        No — and, as usual, the honest answer is the interesting one. The architecture is real: the config for this exact
        model says <code>mtp_num_hidden_layers: 1</code>, so a full Qwen3.5 checkpoint <em>does</em> define the head you
        just saw. But two things keep it off the path in this demo. The bf16 checkpoint the browser downloads was
        converted without the MTP weights — of its 474 tensors, <em>zero</em> are <code>mtp.*</code> — and the WebGPU
        decode loop here is plain autoregressive, one token per pass, with no draft/verify machinery at all. (You are not
        alone: stock Hugging Face Transformers also discards the MTP weights; today it is mainly inference engines like
        vLLM, SGLang, and this project&apos;s own native Metal engine that actually run the loop — measured there at
        roughly <strong>1.06–1.46×</strong> faster, depending on how predictable the text is.)
      </p>
      <p>
        So multi-token prediction sits with the vision encoder and the other parts of the full checkpoint that the
        architecture defines but this browser tour does not walk: real in Qwen3.5, switched off here on purpose. You can
        even see it greyed out in the{' '}
        <a href="/chapters/architecture">architecture map</a> — the &ldquo;MTP head&rdquo; node, off the route a token
        actually travels in this demo.
      </p>
    </Prose>
  );
}

/** Sub-chapter 11.2 — reasoning, self-consistency, and test-time compute. */
export function SamplingReasoningSection() {
  return (
    <Prose>
      <h1>Reasoning &amp; test-time compute</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 11, Sampling — <ChapterLink chapterId="post-training">Chapter 15</ChapterLink> showed you the{' '}
        <code>{'<think>'}</code> pill; here is why making a model &ldquo;think longer&rdquo; before it answers works at
        all, and how it leans on the very sampling knobs from this chapter.
      </p>
      <p>
        A model answering in one shot spends a fixed amount of compute per token, no matter how hard the question is.{' '}
        <strong>Chain-of-thought</strong> (CoT) prompting — Wei et al., 2022 — found something almost embarrassingly
        simple: show a frozen model a few worked examples that write out intermediate steps before the final answer, and
        it starts writing out its <em>own</em> intermediate steps, getting much better at arithmetic, logic, and
        multi-step word problems. No weights changed. An 8-exemplar CoT prompt pushed a 540B PaLM model to a new
        state-of-the-art on GSM8K math problems — beating even a fine-tuned GPT-3 with an external calculator/verifier
        bolted on.
      </p>

      <h2>One chain isn&apos;t enough — vote across many</h2>
      <p>
        CoT gives you one reasoning chain. <strong>Self-consistency</strong> (Wang et al., 2022) asks a sharper
        question: why trust just one? Sample <MathDisplay inline latex={String.raw`n`} /> independent chains at
        temperature <MathDisplay inline latex={String.raw`T > 0`} /> — literally the sampling temperature from earlier in
        this chapter — let each one reach its own final answer, and report whichever answer the most chains agree on:
      </p>
      <MathDisplay latex={String.raw`\hat{y} = \operatorname*{arg\,max}_{a} \sum_{i=1}^{n} \mathbb{1}[\,y_i = a\,]`} />
      <p>
        This is a direct, concrete use of this chapter&apos;s own machinery: at{' '}
        <MathDisplay inline latex={String.raw`T = 0`} /> (greedy) every &ldquo;resample&rdquo; is the identical chain, so
        there is nothing to vote on — self-consistency <em>needs</em> the randomness that temperature sampling supplies to
        produce genuinely different chains in the first place. Wang et al. measured real gains from this alone: +17.9
        points on GSM8K, +11.0 on SVAMP, +12.2 on AQuA, +6.4 on StrategyQA, +3.9 on ARC-challenge, all over greedy
        chain-of-thought. Watch it recover the right answer from a mixed batch of samples, including one that makes the
        exact same mistake greedy decoding would have committed to:
      </p>

      <SelfConsistencyVoting />

      <h2>From a prompting trick to a trained, scalable axis</h2>
      <p>
        Chapter 14&apos;s size ladder plotted accuracy against <em>parameter count</em> — bigger model, more train-time
        compute, better score. In September 2024, OpenAI&apos;s o1 put a second axis on the chart: <strong>test-time
        compute</strong>. o1 is trained with large-scale reinforcement learning to produce a long internal chain of
        thought — billed as output tokens but hidden from the response — before answering, and its accuracy climbs
        log-linearly with <em>both</em> more RL at training time and more thinking tokens at inference time. The jump is
        large: on AIME 2024, GPT-4o scores around 12% versus o1&apos;s 74% with one sample, 83% with a self-consistency
        style consensus over 64 samples, and 93% when re-ranking 1,000 samples — plus 89th-percentile Codeforces and
        above-PhD-level GPQA science scores. Snell et al. (2024) formalized why this works: test-time compute is an
        independent, prompt-adaptive scaling axis, and a <em>compute-optimal</em> way of spending it can match a model{' '}
        <MathDisplay inline latex={String.raw`14\times`} /> larger on problems within the smaller model&apos;s reach,
        beating naive best-of-<MathDisplay inline latex={String.raw`N`} /> sampling by over 4x in compute efficiency.
      </p>

      <h2>Reasoning from pure RL: DeepSeek-R1</h2>
      <p>
        DeepSeek-R1 (Jan 2025) pushed the training side further: DeepSeek-R1-Zero learns to reason — long chains,
        self-reflection, checking its own work — from <strong>pure reinforcement learning</strong> on top of
        DeepSeek-V3-Base, with <em>zero</em> supervised warm-start, using GRPO (Group Relative Policy Optimization) and
        only rule-based accuracy/format rewards, no learned reward model at all. On AIME 2024, pass@1 climbs from 15.6%
        (the base model) to 71.0% after RL — and to 86.7% with the same self-consistency-style majority vote over 64
        samples you just met above, matching OpenAI&apos;s o1-0912. The shipped DeepSeek-R1 adds a small cold-start
        supervised dataset before RL purely to fix readability and language-mixing, reaching 79.8% and o1-parity.
      </p>

      <h2>Doing it on a budget: <code>s1</code> and budget forcing</h2>
      <p>
        You don&apos;t need R1-scale RL to get an R1-shaped curve. The <code>s1</code> paper (Jan 2025) fine-tuned
        Qwen2.5-32B-Instruct on just 1,000 curated examples (s1K) and added <strong>budget forcing</strong>: truncate the
        model&apos;s thinking early to force an answer, or — if it tries to stop too soon — suppress the end-of-thinking
        token and literally append the word &ldquo;Wait&rdquo; to make it keep reasoning. That single lever raised AIME24
        accuracy from 50% to 57% on the same model, and the resulting s1-32B beat o1-preview on competition math by up to
        27 points, no RL required. Notice what budget forcing is actually doing at the token level: directly editing the
        length and content of the <em>same</em> <code>{'<think>...</think>'}</code> block{' '}
        <ChapterLink chapterId="post-training">chapter 15</ChapterLink> showed you being emitted token by token — test-time
        compute made literal as &ldquo;how many tokens do we let the reasoning block run before we force it to stop.&rdquo;
      </p>

      <h2>Does your in-browser Qwen do this?</h2>
      <p>
        Partly, and the gaps are worth naming precisely. Qwen3.5&apos;s own model card discloses only one vague sentence
        about its reasoning training — <em>&ldquo;Scalable RL Generalization: reinforcement learning scaled across
        million-agent environments with progressively complex task distributions&rdquo;</em> — with no GRPO name, no
        reward description, no verifier-pair counts. That is a real step down in disclosure from the predecessor
        architecture&apos;s own Qwen3-Next blog, which published GRPO by name and an exact count — 3,995 query-verifier
        pairs — for its Reasoning-RL stage. What the card <em>does</em> say plainly: the flagship Qwen3.5-397B-A17B
        defaults to <strong>thinking mode on</strong>, but this exact browser&apos;s model, Qwen3.5-0.8B, defaults to{' '}
        <strong>thinking mode off</strong> — the opposite default — and the vendor adds an explicit warning that forcing
        thinking mode on for this smallest checkpoint makes it prone to reasoning loops that fail to terminate. So the
        honest picture is: the same <code>{'<think>'}</code> machinery from{' '}
        <ChapterLink chapterId="post-training">chapter 15</ChapterLink> is real and present in this checkpoint, chain of
        thought and self-consistency are general techniques you could apply to any model including this one by sampling
        it several times yourself, but the vendor&apos;s own advice for the specific 0.8B model running in this tab is to
        leave thinking off.
      </p>
    </Prose>
  );
}

/** Sub-chapter 11.3 — tool calling and constrained/grammar-based decoding. */
export function SamplingToolCallingSection() {
  return (
    <Prose>
      <h1>Tool calling &amp; constrained decoding</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 11, Sampling —{' '}
        <ChapterLink chapterId="post-training">chapter 15</ChapterLink> showed you the <code>tool_call</code> format;
        here is how a production system can make that format <em>unbreakable</em>, using nothing but this chapter&apos;s
        own sampling machinery plus one extra mask.
      </p>
      <p>
        Letting a model call external functions goes back to <strong>ReAct</strong> (Yao et al., Oct 2022): interleave
        free-text reasoning with discrete actions — a lookup, a search — in the same generation stream, so the model can
        plan, act, observe, and re-plan. OpenAI productized the pattern in June 2023 as <strong>function calling</strong>{' '}
        (models <code>gpt-4-0613</code> / <code>gpt-3.5-turbo-0613</code>): the model was fine-tuned to emit a JSON blob
        describing a function call instead of prose. Simon Willison, writing the same week, called it exactly what it was
        — &ldquo;an implementation of the ReAct pattern, with models that have been fine-tuned to execute it.&rdquo; And
        that is the catch: fine-tuning teaches a model to <em>usually</em> emit well-formed JSON, with no hard guarantee
        it always will.
      </p>

      <h2>Masking, not hoping</h2>
      <p>
        The fix doesn&apos;t touch training at all — it happens at the exact moment this chapter has spent the whole
        time on: <strong>picking the next token</strong>. Ordinarily you take the raw logits, maybe apply temperature,
        maybe keep only the top-k or top-p mass, then sample or argmax. <strong>Constrained (grammar-based) decoding</strong>{' '}
        adds one more step before that: track which tokens are still <em>syntactically legal</em> given the schema and
        everything generated so far, and mask every illegal one to <MathDisplay inline latex={String.raw`-\infty`} />{' '}
        before the usual sampling runs. The Outlines paper (Willard &amp; Louf, 2023) formalizes this by compiling a
        regex or JSON-schema/CFG into a finite-state machine and indexing the vocabulary against FSM states; llama.cpp&apos;s
        GBNF grammars do the equivalent job in C++. Walk it step by step over the exact tag format{' '}
        <ChapterLink chapterId="post-training">chapter 15</ChapterLink> introduced —{' '}
        <code>{'<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>'}</code>:
      </p>

      <ConstrainedDecodeMask />

      <p>
        Notice the mask isn&apos;t always the same shape. Mid-value, almost anything is legal (a value is just text) —
        the mask barely narrows the field. Right after <code>get_weather</code>&apos;s one required parameter closes,
        exactly <em>one</em> token is legal, <code>{'</function>'}</code> — the mask alone decides the next token before
        temperature or top-p even get a turn. That is the general lesson: grammar masking guarantees the <em>syntax</em>{' '}
        is unbreakable; it says nothing about whether the <em>content</em> — the city name, the reasoning behind it — is
        any good. Those two are genuinely separate problems.
      </p>

      <h2>What masking buys you, measured</h2>
      <p>
        OpenAI&apos;s own numbers make the case cleanly. When they shipped <strong>Structured Outputs</strong> (
        <code>strict: true</code>) in August 2024, they described it as moving from &ldquo;fine-tune and hope&rdquo; to
        &ldquo;a deterministic, engineering-based approach called constrained sampling.&rdquo; On their complex
        JSON-schema evaluation: <code>gpt-4-0613</code> (2023-era function calling, fine-tuning only) scored under 40%;
        a newer model, <code>gpt-4o-2024-08-06</code>, with fine-tuning alone reached 93%; adding constrained decoding
        (Structured Outputs) on top reached{' '}
        <strong>100%</strong>. Masking doesn&apos;t make the model smarter about <em>what</em> to call — it makes the{' '}
        <em>shape</em> of the call unbreakable.
      </p>

      <h2>Does your in-browser Qwen do this?</h2>
      <p>
        No. This repository&apos;s sampler (<code>crates/mlx-core/src/sampling.rs</code>) implements exactly temperature,
        top-k, top-p, and min-p — the knobs from earlier in this chapter — and nothing that masks tokens for grammar or
        schema validity: no <code>logit_bias</code>, no allowed-token mask, no compiled grammar anywhere in the pipeline
        (confirmed by reading the sampler source and grepping this project&apos;s browser/WASM driving code for
        grammar-shaped keywords — none exist). Instead, tool calls are recovered <em>after the fact</em>: the model
        generates completely freely, and <code>crates/mlx-core/src/tools/mod.rs</code> — its own doc comment says so
        directly — &ldquo;uses simple string-based parsers instead of regex&rdquo; to scan the finished text for{' '}
        <code>{'<tool_call>'}</code> tags once generation is done. The clearest proof there is no hard constraint: the
        parser&apos;s own result type, <code>ToolCallResult</code>, has real failure states —{' '}
        <code>invalid_json</code>, <code>missing_name</code>, <code>parse_error</code> — that only exist because nothing
        stopped the model from producing malformed output in the first place. A true grammar mask would make those states
        unreachable by construction; here, they are live code paths. So the tag format{' '}
        <ChapterLink chapterId="post-training">chapter 15</ChapterLink> showed you is exactly as trustworthy as the
        model&apos;s own learned habit of formatting it correctly — the same category as OpenAI&apos;s original 2023
        function calling, not the constrained-decoding category that came after it.
      </p>
    </Prose>
  );
}
