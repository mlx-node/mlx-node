// Sub-chapter bodies for chapter 12 (KV cache & hybrid attention).
//
// These are the "go deeper" deep-dives that live at their own sub-routes
// (/chapters/kv-cache/<id>) instead of inline in the chapter. They are pure,
// model-free reading prose — SSR-safe, so the prerenderer renders them straight
// to crawlable static HTML (see scripts/prerender.entry.tsx). Each is wrapped in
// <Prose> by the section route / prerenderer, so it returns just the article
// content (leading <h1> + paragraphs).
//
// The framing is grounded in the real serving picture, and is honest about the
// in-browser demo: it runs as a SINGLE user at batch size 1 — the pathological
// worst case for the memory-bound decode the course already taught — with no
// cross-request prefix to share. Production serving batches many users and reuses
// a fixed system prompt; those are exactly the levers these two sub-chapters
// explain. Every metric/number is illustrative teaching shape, not a measurement
// of this model (the widgets repeat that caveat).

import * as React from 'react';

import { Prose } from '../Prose';
import { BatchingModesDiagram } from '../widgets/BatchingModesDiagram';
import { ChunkedPrefillEngine } from '../widgets/ChunkedPrefillEngine';
import { DynamicQuantMix } from '../widgets/DynamicQuantMix';
import { FormatZooLadder } from '../widgets/FormatZooLadder';
import { Fp8Fork } from '../widgets/Fp8Fork';
import { GgufQuantLadder } from '../widgets/GgufQuantLadder';
import { GranularityZoom } from '../widgets/GranularityZoom';
import { InferenceMetricsDiagram } from '../widgets/InferenceMetricsDiagram';
import { MxBlockAnatomy } from '../widgets/MxBlockAnatomy';
import { NF4BellCurve } from '../widgets/NF4BellCurve';
import { NumberLineGrid } from '../widgets/NumberLineGrid';
import { PrefixCacheDiagram } from '../widgets/PrefixCacheDiagram';
import { QuantizationDiagram } from '../widgets/QuantizationDiagram';
import { QuantizeRoundTrip } from '../widgets/QuantizeRoundTrip';
import { QuantMethodsMap } from '../widgets/QuantMethodsMap';
import { RangePrecisionScatter } from '../widgets/RangePrecisionScatter';
import { SixteenBitFork } from '../widgets/SixteenBitFork';
import { SymmetricAsymmetric } from '../widgets/SymmetricAsymmetric';

/** Sub-chapter 12.1 — how serving speed is measured, and how batching multiplies it. */
export function KvCacheBatchingSection() {
  return (
    <Prose>
      <h1>Latency, throughput &amp; batching</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 12, KV cache — how &ldquo;fast&rdquo; is actually measured, and how a server turns one
        memory-bound decode into throughput for many.
      </p>
      <p>
        The KV cache is <em>why</em> decode is fast: each new token reuses the keys and values it already computed
        instead of redoing the whole prompt. But &ldquo;fast&rdquo; is two different numbers depending on what you mean,
        and a real server has to serve not one user but hundreds at once. This sub-chapter is the serving layer: first
        how speed is <em>measured</em>, then how <strong>batching</strong> turns the memory-bound decode you watch in
        the demo into throughput.
      </p>

      <h2>Two clocks — and why the average lies</h2>
      <p>
        When output is streamed token-by-token, two clocks matter and they measure different things.{' '}
        <strong>TTFT</strong> (time to first token) is the pause before the first word appears — it&apos;s set by the{' '}
        <em>compute-bound</em> prefill digesting your whole prompt in one pass. <strong>TPOT</strong> (time per output
        token, also called <strong>ITL</strong>, inter-token latency) is the steady streaming speed after that — set by
        the <em>memory-bound</em> decode. They trade off and you tune them separately: an ITL of <code>10&nbsp;ms</code>{' '}
        is <code>100 tokens/sec</code> for that one user. (For a <em>non-streamed</em> call — an agent making a tool
        call and waiting for the whole JSON — neither clock is what you feel; you measure total response time instead.)
      </p>
      <p>
        The trap is reporting either one as an <em>average</em>. Inference latency is <strong>right-skewed</strong>:
        most requests cluster near the typical case, with a long slow tail of unlucky ones. The mean gets dragged
        rightward by that tail and <em>hides</em> it — so engineers report <strong>percentiles</strong> instead. P50 is
        the median (half are slower), P90 is one-in-ten, P99 is the worst one in a hundred. Reliability work targets{' '}
        <strong>P90/P99</strong>, because the tail is what users actually remember. Watch both clocks, and watch the
        average lie:
      </p>

      <InferenceMetricsDiagram />

      <h2>Batching — one weight-read, many tokens</h2>
      <p>
        Now the lever. Recall the <a href="/chapters/attention/roofline">roofline</a>: a single decode step streams{' '}
        <em>all</em> of the model&apos;s weights from memory exactly once, and at batch 1 that entire sweep produces a
        single token — the far-left memory cliff. The fix is almost embarrassingly direct. Put <code>N</code> requests
        on that <em>same</em> weight-read and the GPU emits <code>N</code> tokens for one memory sweep. Total throughput
        climbs with the batch while the memory traffic stays flat — batching is the cure for the memory wall, and the
        way a decode workload climbs the roofline toward the ridge.
      </p>
      <p>
        Servers differ in <em>when</em> they launch a batch. <strong>Static</strong> batching waits until the batch is
        full, so the first arrival pays for the last. <strong>Dynamic</strong> batching launches when it&apos;s full{' '}
        <em>or</em> a timeout fires, whichever comes first. <strong>Continuous</strong> (a.k.a. <em>in-flight</em>)
        batching — what <code>vLLM</code> and <code>SGLang</code> do — swaps new requests into free slots at the{' '}
        <em>token</em> level, so each one starts almost immediately and the queue stays tiny. Toggle the three and watch
        the request lanes re-time:
      </p>

      <BatchingModesDiagram />

      <p>
        The catch is that batching is a <strong>knob, not a free win</strong>. Slide the batch up and two readouts move
        in <em>opposite</em> directions: total throughput rises and then saturates (you leave the memory-bound regime
        and hit a compute ceiling), while each individual user&apos;s latency rises (more lane-mates to share every step
        with). Picking a batch size is choosing a point on that latency-versus-throughput curve — fast for one user, or
        cheap per token across many.
      </p>
      <p className="text-muted-foreground">
        Which is exactly why this in-browser tour feels different from a hosted API: you are a{' '}
        <strong>single user at batch 1</strong>, the worst case for a memory-bound decode, because that one expensive
        weight-read carries exactly one token. A shared production server weaves dozens of strangers into one continuous
        batch and amortizes the same read across all of them — same model, opposite economics.
      </p>

      <h2>Chunked prefill: how a long prompt stops hogging the GPU</h2>
      <p>
        Continuous batching has a problem we glossed over. If a new request arrives with a{' '}
        <strong>very long prompt</strong>, its prefill is one big compute-bound pass — and while that pass runs, every
        other request in flight gets <em>nothing</em>. Their decode stalls. One user pasting a novel freezes the tokens
        of everyone else on the server.
      </p>
      <p>
        The fix is <strong>chunked prefill</strong>: give each engine step a <em>token budget</em> and split a long
        prompt across as many steps as it takes. Each chunk still writes its keys and values into the cache, so nothing
        is recomputed — and because a chunk leaves room in the budget, other requests&apos; decode steps ride along in
        the very same forward pass. Watch one prompt go through, step by step:
      </p>

      <ChunkedPrefillEngine />

      <p>
        Two details in that animation matter more than they look. First, the <strong>KV cache is paged</strong> — fixed
        size blocks filled left to right, with the last one usually part-empty. That leftover is the only waste, and
        it&apos;s capped at one block per sequence, instead of reserving a contiguous worst-case slab up front. Second,
        the <strong>first token appears only after the final chunk</strong>: the logits that matter live at the last
        prompt position, so every earlier chunk is pure cache-population — real work, but silent.
      </p>
    </Prose>
  );
}

/** Sub-chapter 12.2 — reuse one prompt's KV cache across requests, not just within one. */
export function KvCachePrefixCachingSection() {
  return (
    <Prose>
      <h1>Prefix caching</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 12, KV cache — reuse one prompt&apos;s KV <em>across</em> requests, not just within a single
        generation.
      </p>
      <p>
        The KV cache you met in this chapter is reused <em>within</em> one generation: each token of a single reply
        reuses the keys and values of the tokens before it. <strong>Prefix caching</strong> takes the same idea one
        level up — reuse that KV <em>across separate requests</em> that happen to begin with the same tokens.
      </p>
      <p>
        The mechanism is exact. Two requests that share a leading <strong>prefix</strong> share the keys and values for
        those tokens, so the prefix is computed <strong>once</strong> and every later request <em>skips prefill</em>{' '}
        over it — a <strong>cache hit</strong> that drops its TTFT straight to the first new token. But the reuse ends
        at the <strong>first token that differs</strong>. The model is autoregressive, so one different token changes
        the hidden state — and therefore the KV — for <em>everything</em> after it, even text that looks identical
        further down. Past the first miss, the cache is worthless:
      </p>

      <PrefixCacheDiagram />

      <p>
        That first-difference rule is also a design lesson: <strong>put novel tokens as late as you can</strong>. A long
        fixed <strong>system prompt</strong> followed by the user&apos;s question reuses the whole system prompt every
        time; the same content with a per-request id pasted at the <em>front</em> reuses nothing. The big wins are
        exactly the high-reuse shapes — long system prompts for agents and chatbots, <strong>RAG</strong> with shared
        retrieved documents, shared code context, and multi-turn chat (which re-sends the whole conversation so far on
        every turn).
      </p>
      <p className="text-muted-foreground">
        And the honest footnote for this demo: as a single user at batch 1 you have nothing to share a prefix{' '}
        <em>with</em> — every request re-prefills the whole prompt cold. A production API reuses one fixed system prompt
        across millions of requests, which is why pay-per-token APIs bill <strong>cache-hit</strong> tokens cheaper and
        feel near-instant on the boilerplate, only &ldquo;thinking&rdquo; on the tokens that are actually new.
      </p>
    </Prose>
  );
}

/** Sub-chapter 12.3 — fewer bits per weight: the most direct lever on the memory-bound decode. */
export function KvCacheQuantizationSection() {
  return (
    <Prose>
      <h1>Quantization</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 12, KV cache — store each weight in fewer bits, and the same memory-bound decode runs
        lighter and faster.
      </p>
      <p>
        Everything in this chapter pointed at one bottleneck: decode is <strong>memory-bound</strong>. Every token
        re-reads the entire model from memory, and at batch 1 that whole sweep buys a single token.{' '}
        <a href="/chapters/attention/roofline">The roofline</a> put a number on it — you are pinned to the far-left
        memory cliff. <strong>Quantization</strong> is the most direct lever on that cliff, and it leaves the
        architecture untouched: same shapes, same forward pass — it just <em>stores each weight in fewer bits</em>, an
        approximate copy on a coarser grid. Fewer bytes per weight means less to stream, and the memory-bound decode
        runs lighter.
      </p>

      <h2>Fewer bits, smaller model, faster decode</h2>
      <p>
        A weight in <code>bf16</code> — what this demo ships — takes 2 bytes. Drop to <code>fp8</code> or{' '}
        <code>int8</code> and it&apos;s 1 byte; <code>int4</code> is half a byte. The footprint scales straight down
        with the bits, and because decode is memory-bound, so does the time to stream those weights. But it is{' '}
        <strong>not</strong> a clean 2× per level: overhead — dequantizing back to a compute format, imperfect bandwidth
        use — means the book quotes a more honest <strong>30–50% speedup per precision level down</strong>. Step through
        the formats and watch all three bars — size, speed, quality — move together:
      </p>

      <QuantizationDiagram />

      <h2>How it works — and what it costs</h2>
      <p>
        The trick is a <strong>scale factor</strong>: a block of weights is mapped onto a small grid of integer levels,
        and one shared float rescales them back at compute time (
        <span className="font-mono">real ≈ scale × (q − zero_point)</span> — where <em>zero_point</em> just shifts the
        grid so real 0 still lands on an exact code; more on this in the{' '}
        <a href="/chapters/kv-cache/number-formats">number-formats sub-chapter</a>). Fewer bits means a coarser grid, so
        the cost is <strong>quality</strong>. The book ranks what tolerates it, least to most fragile:{' '}
        <strong>weights &lt; activations &lt; KV cache &lt; attention</strong>. Weight-only methods like{' '}
        <strong>GPTQ</strong> and <strong>AWQ</strong> quantize just the weights and are often near-lossless down to{' '}
        <code>int4</code>; pushing to weights-<em>and</em>-activations (<code>W8A8</code>, e.g.{' '}
        <strong>SmoothQuant</strong>, <strong>LLM.int8()</strong>) is harder because activations carry wilder outliers.
        The rule of thumb: <code>fp8</code> / MXFP8 is the sweet spot — almost no perceptible loss — while integer
        formats lack dynamic range, so for quality-sensitive work you stick with floating point and treat sub-4-bit as
        the cliff.
      </p>

      <h2>The KV cache can be quantized too</h2>
      <p>
        This isn&apos;t only about weights. The <strong>bf16 KV cache</strong> you met in this chapter is itself a
        stream of bytes the decode reads back every step — so it&apos;s a quantization target as well. It&apos;s{' '}
        <em>moderately</em> sensitive (the book&apos;s order again): its errors <strong>compound token to token</strong>
        , since each cached key/value is reused for the entire rest of the sequence. <code>fp8</code> / MXFP8 is the
        safe choice there; aggressive integer KV quant is where quality starts to slip.
      </p>
      <p className="text-muted-foreground">
        Honest footnote for this demo: the in-browser model runs in <strong>bf16</strong> with{' '}
        <strong>no quantization at all</strong> — the simplest, highest-fidelity choice. The book notes integer /
        sub-4-bit quant is mostly a <em>local-inference</em> practice (the GGUF files you&apos;d download for Ollama or
        llama.cpp), distinct from datacenter serving. A quantized copy of this same model would be ≈2× (
        <code>int8</code>) to ≈4× (<code>int4</code>) smaller and a real chunk faster to decode — the lever this whole
        chapter has been pointing at.
      </p>
    </Prose>
  );
}

/**
 * Sub-chapter 12.4 — Number formats & precision. The deep companion to the
 * quantization teaser above: how a weight is actually stored as bits, the
 * precision AND dynamic range of every common format (fp32 → fp4, int8/int4,
 * nf4, the MX/NVFP4 block formats), and how quantization snaps a continuous
 * weight onto a discrete grid (scale, zero-point, granularity, calibration,
 * the GPTQ/AWQ/SmoothQuant/LLM.int8()/GGUF/NF4 method zoo). Ten diagrams carry
 * the visual load; every number is primary-source-verified (IEEE 754-2019; the
 * OCP OFP8 & Microscaling MX v1.0 specs; the FP8 paper arXiv:2209.05433; the
 * Nagel et al. quantization white paper arXiv:2106.08295; GPTQ 2210.17323, AWQ
 * 2306.00978, SmoothQuant 2211.10438, LLM.int8() 2208.07339, QLoRA/NF4
 * 2305.14314; NVIDIA Ampere/Hopper/Blackwell whitepapers). Quality stays
 * qualitative — the sources give no perplexity numbers, so neither do we.
 */
export function KvCacheNumberFormatsSection() {
  return (
    <Prose>
      <h1>Number formats &amp; precision</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 12, KV cache — open the hood on quantization: how a weight is stored as bits, what every
        format can and cannot represent, and how a real number gets snapped onto a coarse grid.
      </p>
      <p>
        The <a href="/chapters/kv-cache/quantization">previous sub-chapter</a> used quantization as a lever: fewer bits
        per weight, less to stream, a faster memory-bound decode. This one opens the hood. What <em>is</em> a weight,
        down at the bits? Why does <code>bf16</code> hold numbers as large as 10³⁸ while <code>fp16</code> overflows
        just above 65,504? And what exactly happens when you &ldquo;store a weight in fewer bits&rdquo;? It comes down
        to two ideas — how a float splits its bits, and how a grid stands in for a continuous number.
      </p>

      <h2>A weight is just bits</h2>
      <p>
        A floating-point number is <span className="font-mono">sign × mantissa × 2^exponent</span>, and the bits split
        into those three fields. The split is the whole game: the <strong>exponent</strong> field buys{' '}
        <strong>dynamic range</strong> (how big and how small a value can get), and the <strong>mantissa</strong> field
        buys <strong>precision</strong> (how finely values are spaced). An integer has no exponent at all — it is a{' '}
        <strong>uniform</strong> grid, every step the same size, with no dynamic range of its own. That one contrast is
        the foundation of everything below:
      </p>
      <NumberLineGrid />
      <p>
        Integers spread their levels evenly across the range. Floats cluster them near zero — where most weights
        actually live — and let them spread out toward the maximum. Same count of levels, opposite layout, and only the
        float carries any range. Spend a bit on the exponent and you can reach 10³⁸; spend it on the mantissa and you
        resolve a finer difference. You cannot have both for free.
      </p>

      <h2>The same 16 bits, two ways</h2>
      <p>
        The cleanest illustration is the two 16-bit formats. They have the <em>same</em> bit budget and split it
        oppositely — and that single choice decides what each is good for:
      </p>
      <SixteenBitFork />
      <p>
        <code>fp16</code> keeps a big mantissa (fine precision) but a small 5-bit exponent, so it tops out at 65,504 and
        underflows easily — training in <code>fp16</code> needs loss scaling to keep gradients from vanishing.{' '}
        <code>bf16</code> keeps fp32&apos;s full 8-bit exponent, so it matches fp32&apos;s enormous range (just with
        coarser steps) and rarely needs loss scaling. That is why <code>bf16</code> became the default training
        precision — and it is exactly what <strong>this demo ships</strong>. You are here.
      </p>

      <h2>The format zoo</h2>
      <p>
        Below 16 bits the field split becomes a real design decision, so each bit-width ships in more than one flavor.
        Here is the whole ladder — click any row to read what it can represent:
      </p>
      <FormatZooLadder />
      <p>
        Notice <code>fp8</code> comes in two: <strong>E4M3</strong> (more mantissa, max 448 — for weights and
        activations) and <strong>E5M2</strong> (more exponent, max 57,344 — for gradients). It is the same fork as the
        16-bit pair, one tier down — and fp8 training uses <em>both</em> halves at once:
      </p>
      <Fp8Fork />
      <p>
        Plot every format on the range-versus-precision plane and the reason two exist per width is obvious: at a fixed
        bit-width you have to pick a <em>corner</em>.
      </p>
      <RangePrecisionScatter />
      <p>
        A footgun for fact-checkers: <code>fp8</code> E4M3&apos;s max of 448 is the ML (OCP) convention — a naive
        IEEE-style 8-bit float would stop at 240. E4M3 reclaims the infinity and most NaN bit-patterns to extend its
        range, and the tiny <code>fp6</code>/<code>fp4</code> formats drop infinities and NaNs entirely to win back a
        few precious code points. <strong>Subnormals</strong> (gradual underflow, where the implicit leading 1 becomes
        0) buy a handful of extra tiny magnitudes at reduced precision — the smallest numbers each format can still
        name.
      </p>

      <h2>Snapping reals onto a grid</h2>
      <p>
        Quantizing a weight is just rounding it to the nearest tick of a grid. The grid spacing is the{' '}
        <strong>scale factor</strong> <span className="font-mono">s</span>; whatever is left over is the quantization
        error, at most half a step:
      </p>
      <QuantizeRoundTrip />
      <p>
        That is the entire mechanic: <span className="font-mono">q = round(x / s)</span> picks the nearest level,{' '}
        <span className="font-mono">x̂ = s · q</span> reads it back, and the error never exceeds{' '}
        <span className="font-mono">s / 2</span>. A coarser grid (fewer bits) means a bigger step and more error. But
        weights are not always centered on zero — and that is what the <strong>zero-point</strong> handles:
      </p>
      <SymmetricAsymmetric />
      <p>
        Weights are roughly zero-mean, so they get a <strong>symmetric</strong> grid (zero-point 0). Activations are
        skewed and one-sided (think post-ReLU, all ≥ 0), so they get an <strong>asymmetric</strong> grid — the
        zero-point shifts the whole lattice so that real 0 still lands exactly on an integer code (
        <span className="font-mono">q = round(x / s) + z</span>). This is the deeper reason weight-only quantization is
        the easy first step and quantizing activations (<code>W8A8</code>) is the hard part: activations carry wild,
        input-dependent outliers that a single scale cannot hold.
      </p>

      <h2>One scale for how many weights?</h2>
      <p>
        A single scale for an entire tensor is cheap to store but loose — one outlier weight stretches the grid for
        everyone. Give each output channel, or each block of 32–128 weights, its own scale, and every grid fits its own
        values more tightly. That is the <strong>granularity</strong> ladder:
      </p>
      <GranularityZoom />
      <p>
        Finer granularity means lower error — but the scales cost bits too, so a &ldquo;4-bit&rdquo; weight with a
        shared scale every 64 values is really about <strong>4.25 bits</strong>. The Microscaling (MX) standard freezes
        this at a 32-element block, which is the bridge to the smallest formats.
      </p>

      <h2>Block formats borrow the range back</h2>
      <p>
        On its own, a 4-bit float can name only a handful of magnitudes —{' '}
        <span className="font-mono">{'{0, 0.5, 1, 1.5, 2, 3, 4, 6}'}</span>. That is useless until a block of them{' '}
        <em>shares</em> one scale that supplies the magnitude. This is why <code>fp4</code> and <code>fp6</code> are{' '}
        <em>only ever</em> block formats — and the same microscaling recipe also wraps <code>fp8</code>. Toggle the
        three the course names:
      </p>
      <MxBlockAnatomy />
      <p>
        The OCP <strong>MX</strong> family applies one rule at three widths, always with a power-of-two{' '}
        <code>E8M0</code> scale (8 exponent bits, no mantissa, spanning 2⁻¹²⁷ … 2¹²⁷) shared across a 32-element block.{' '}
        <strong>MXFP8</strong> wraps <code>fp8</code> at ≈ <strong>8.25</strong> bits/elem — fp8 already has range, so
        this is the near-lossless end. <strong>MXFP4</strong> applies the same block to <code>fp4</code> at ≈{' '}
        <strong>4.25</strong> bits/elem, where the shared scale is what makes 4-bit usable at all. NVIDIA&apos;s{' '}
        <strong>NVFP4</strong> tightens the block to 16 elements, swaps in a richer <code>fp8</code> E4M3 block scale (a
        real float, with a mantissa, finer than a power of two), and adds a second per-tensor <code>fp32</code> scale (≈{' '}
        <strong>4.50</strong> bits/elem) — two levels of scaling to claw back both range and accuracy.
      </p>

      <h2>The clever 4-bit: NF4</h2>
      <p>
        If pretrained weights are roughly bell-shaped, why space the levels evenly at all? <strong>NF4</strong> does
        not. It places its 16 levels as the <em>quantiles</em> of a normal distribution — dense where the weights
        actually sit, sparse out in the tails:
      </p>
      <NF4BellCurve />
      <p>
        Uniform <code>int4</code> spends as many of its 16 codes on the rare tails as on the crowded center; NF4 follows
        the data, so more of its resolution lands where the weights are. It is information-theoretically optimal for
        normally-distributed weights, includes an exact 0, and is really a 16-entry <em>lookup table</em> rather than a
        computed grid — the 4-bit base that <strong>QLoRA</strong> finetunes on top of.
      </p>

      <h2>Picking a recipe</h2>
      <p>
        Those pieces — a format, a scale, a granularity — combine into named methods. They sort onto two axes: how much
        you quantize (weights only, or weights <em>and</em> activations), and how much retraining you are willing to do
        (<strong>PTQ</strong>, just calibrate; or <strong>QAT</strong>, retrain through fake-quant):
      </p>
      <QuantMethodsMap />
      <p>
        For open weights you almost always do PTQ — QAT needs a full training loop. Most local-inference quant (
        <strong>GPTQ</strong>, <strong>AWQ</strong>, <strong>GGUF</strong>, <strong>NF4</strong>) is weight-only: it
        shrinks memory and speeds the memory-bound decode without touching activations. Two traps worth naming:{' '}
        <strong>AWQ</strong> is weight-only <em>despite</em> the &ldquo;Activation-aware&rdquo; name (activations only{' '}
        <em>pick</em> which weight channels to protect), and a GGUF <code>Q4_K</code> is about <strong>4.5</strong> bits
        per weight, not 4, once you count its stored block scales (other <code>Q4*</code> variants land elsewhere).
      </p>

      <h2>The one you actually download: GGUF k-quants</h2>
      <p>
        On Hugging Face, Ollama, or LM Studio you do not pick &ldquo;int4&rdquo; — you pick a{' '}
        <strong>GGUF Q-type</strong>. These are the llama.cpp file formats, and the &ldquo;<code>_K</code>&rdquo; ones
        are <em>k-quants</em>: a super-block of 256 weights with a two-level scale — one or two super-block floats (a
        second, <code>dmin</code>, for types with an asymmetric zero-point) plus a small quantized scale per sub-block
        of 16 or 32 weights, depending on the exact type. That is the same block-format trick you just saw in MX, which
        is exactly why their cost is a <em>fractional</em> bits-per-weight, never a round number. Each rung below is the{' '}
        <strong>pure-type</strong> bits/weight, read straight off the C struct in <code>ggml-common.h</code>:
      </p>
      <GgufQuantLadder />
      <p>
        Two honest traps. First, the names everyone actually downloads — <code>Q4_K_M</code>, <code>Q3_K_M</code>,{' '}
        <code>Q5_K_M</code> — are not pure types but <strong>mixes</strong>: the <code>_M</code> keeps the most
        sensitive tensors higher — always <code>attn_v</code> and <code>ffn_down</code>, plus <code>attn_o</code> in{' '}
        <code>Q3_K_M</code> — so the <em>file</em> lands a bit above the pure number (an 8B <code>Q4_K_M</code> is ≈{' '}
        <strong>4.9</strong> bpw, not 4.5). Second, the <code>IQ</code> rungs (<strong>I-quants</strong>) are not
        rounded grids at all — each weight is an index into a fixed codebook, chosen with the help of an{' '}
        <strong>importance matrix</strong>: run a little calibration text through the model, see which weights move the
        output most, and spend the bits there. Below ~3 bits that imatrix is what makes the file usable at all.
      </p>
      <p>
        Generalize that &ldquo;keep the sensitive tensors high&rdquo; idea to <em>every</em> layer and you get{' '}
        <strong>dynamic</strong> quantization — the heart of Unsloth&apos;s Dynamic 2.0 GGUFs. Same size budget, spent
        where calibration says it matters:
      </p>
      <DynamicQuantMix />
      <p>
        Unsloth&apos;s <strong>Dynamic 2.0</strong> picks the quant <em>type per layer, per model</em> from a &gt;
        1.5M-token calibration set, and grades the result by <strong>KL-divergence</strong> to the original rather than
        perplexity (perplexity can cancel its own errors out — see &ldquo;Accuracy is Not All You Need,&rdquo;{' '}
        <span className="font-mono">arXiv 2407.09141</span>). Pushed to the floor, the same idea squeezed a 671B
        DeepSeek-R1 from ~720&nbsp;GB down to <strong>131&nbsp;GB</strong> at a ~1.58-bit <code>IQ1</code> average —
        keeping the few dense and attention layers at 4–6 bit while the MoE experts drop to ~1.5. One caveat worth
        keeping straight: that &ldquo;1.58-bit&rdquo; is <em>post-training</em> dynamic quantization, not the same thing
        as <strong>BitNet b1.58</strong> (<span className="font-mono">arXiv 2402.17764</span>), which <em>trains</em>{' '}
        ternary weights {'{−1, 0, 1}'} from scratch — they just share the number log₂3 ≈ 1.58.
      </p>

      <h2>Where the hardware draws the line</h2>
      <p>
        Each format exists because the previous one ran out of range or precision for some workload — and the silicon
        followed. NVIDIA <strong>Turing</strong> (2018) first added <code>int8</code>/<code>int4</code> Tensor Cores for
        inference; <strong>Ampere</strong> (A100) added <code>tf32</code>/<code>bf16</code>/<code>fp64</code> Tensor
        Core support and structured sparsity; <strong>Hopper</strong> (H100) added native <code>fp8</code> (E4M3/E5M2);{' '}
        <strong>Blackwell</strong> (B200) added <code>fp4</code> (NVFP4/MXFP4), with Tensor-Core throughput roughly{' '}
        <em>doubling</em> each time the precision halves. Apple-Silicon GPUs &mdash; and consumer cards before
        NVIDIA&rsquo;s Blackwell generation (the RTX 50-series) &mdash; lack native sub-8-bit <em>float</em> units, so
        low-bit inference there leans on <code>int4</code>/<code>int8</code> integer math — which is exactly why
        aggressive integer quant is a local-inference story, not a datacenter one.
      </p>
      <p className="text-muted-foreground">
        Honest footnote, same as the teaser: this browser tab ships <strong>bf16</strong> for <em>both</em> the weights
        and the KV cache — no quantization at all, the highest-fidelity choice. Everything on this page is the menu of
        what you <em>could</em> trade that fidelity for: a smaller, faster copy of the same model, bought with a coarser
        grid. You are here, at the top of the ladder, looking down.
      </p>
    </Prose>
  );
}
