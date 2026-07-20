// Sub-chapter bodies for chapter 4 (Self-attention).
//
// These are the "go deeper" deep-dives that live at their own sub-routes
// (/chapters/attention/<id>) instead of inline in the chapter. They are pure,
// model-free reading prose — SSR-safe, so the prerenderer renders them straight
// to crawlable static HTML (see scripts/prerender.entry.tsx). Each is wrapped in
// <Prose> by the section route / prerenderer's <LessonLayout>, so it returns
// just the article content (leading <h1> + paragraphs).
//
// Runtime claims are grounded in the real WebGPU backend: the occupancy gate in
// mlx/backend/webgpu/sdpa.cpp (~:1268-1277) routes single-token decode (B·H=8 <
// 32) to the decomposed matmul→softmax→matmul path and reserves the fused,
// tiled FlashAttention-2 kernel for prefill; only 6 of 24 layers are full
// softmax attention (the other 18 are GatedDeltaNet linear).

import * as React from 'react';

import { Prose } from '../Prose';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { Flash2ParallelismDiagram } from '../widgets/Flash2ParallelismDiagram';
import { Flash3AsyncDiagram } from '../widgets/Flash3AsyncDiagram';
import { FlashAttentionTimeline } from '../widgets/FlashAttentionTimeline';
import { FlashTilingDiagram } from '../widgets/FlashTilingDiagram';
import { FlashTrafficDiagram } from '../widgets/FlashTrafficDiagram';
import { Fp8QuantDiagram } from '../widgets/Fp8QuantDiagram';
import { GpuHierarchyDiagram } from '../widgets/GpuHierarchyDiagram';
import { MemoryWallDiagram } from '../widgets/MemoryWallDiagram';
import { NativeAttentionDiagram } from '../widgets/NativeAttentionDiagram';
import { RooflineDiagram } from '../widgets/RooflineDiagram';
import { StreamingSoftmaxDiagram } from '../widgets/StreamingSoftmaxDiagram';

/** Sub-chapter 4.1 — why the N×N score matrix is the cost center. */
export function AttentionMemoryWallSection() {
  return (
    <Prose>
      <h1>The memory wall</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 4, Self-attention — how big <code>softmax(QKᵀ)</code> actually gets.
      </p>
      <p>
        Look again at <code>Q · Kᵀ</code>: it is a <code>[seq_len, seq_len]</code> matrix — one score for every pair of
        tokens. For a 4,096-token prompt that is ~16.8 million scores <em>per head, per layer</em>. With 8 query heads
        that is ~134 million numbers for a single attention layer — about 268 MB in bf16 (2 bytes each). The model never
        needs all of that at once, yet the naïve recipe builds the whole thing and sets it aside.
      </p>

      <h2>First, two kinds of GPU memory</h2>
      <p>
        To see why that hurts, you need one fact about how a GPU is built. It has <em>two</em> very different places to
        keep numbers:
      </p>
      <ul>
        <li>
          <strong>SRAM</strong> — a sliver of memory right next to the compute units, where the actual math happens. It
          is blazing fast but <em>tiny</em> (a few megabytes). Think of the small desk you work at.
        </li>
        <li>
          <strong>HBM</strong> — the GPU&apos;s main memory, where weights and activations live. It is large (gigabytes)
          but comparatively far away and slow to reach. Think of a warehouse down the hall.
        </li>
      </ul>
      <p>
        Anything too big for the desk has to be stored in the warehouse and carried back whenever you need it. The N×N
        score matrix is far too big for SRAM — so it lives in HBM, and every step that touches it pays for the trip.
      </p>

      <h2>The cost is bandwidth, not storage</h2>
      <p>
        <em>Bandwidth</em> is how <em>fast</em> you can move data between HBM and the compute units — not how much fits.
        The naïve attention recipe is brutal on it: it writes the full N×N matrix out to HBM, reads it back to run
        softmax, writes the normalized scores back, then reads them a third time to weight <code>V</code>. The matrix
        crosses the slow memory bus about four times, and the arithmetic in between is comparatively trivial. Attention
        spends most of its time <em>shuttling that matrix</em>, not computing — so we call it memory-bandwidth-bound.
      </p>
      <p>Watch it happen — and drag the sequence length up to feel the N² blow-up:</p>

      <MemoryWallDiagram />

      <p className="text-muted-foreground">
        One caveat for honesty: this only bites during <em>prefill</em> — processing a long prompt&apos;s positions in
        parallel, whether that takes one pass or several chunked ones — and only on the 6 full-attention layers.
        Single-token decode (one new query row, the part you watch in the chapter demo) and the 18 GatedDeltaNet layers
        never build an N×N matrix at all.
      </p>
      <p>
        That is exactly the problem the next sub-chapter sidesteps: computing the same answer without ever writing the
        N×N matrix to HBM.
      </p>
    </Prose>
  );
}

/** Sub-chapter 4.2 — Flash Attention, online softmax, and what Qwen actually runs. */
export function AttentionFlashAttentionSection() {
  return (
    <Prose>
      <h1>Flash Attention</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 4, Self-attention — the same result, without the N×N matrix.
      </p>
      <p>
        <strong>Flash Attention</strong> computes the same <code>softmax(QKᵀ/√d)·V</code> — the <em>exact</em> attention
        operation, not an approximation like sparse or low-rank attention — without ever materializing the N×N matrix.
        It tiles <code>Q</code>, <code>K</code>, and <code>V</code> into small blocks and, for each query tile, streams
        over the key tiles while keeping a small running state per query row: two single numbers — the max score so far{' '}
        <code>m</code> and the running sum of exponentials <code>ℓ</code> — plus a running output <em>vector</em>{' '}
        <code>o</code>, one entry per output dimension.
      </p>
      <p className="text-muted-foreground">
        &ldquo;Exact&rdquo; here means the algorithm, not the bits: because it sums the key tiles in a different order,
        the output matches the naïve path up to floating-point rounding, not bit-for-bit. That is the normal
        floating-point caveat — reordering a sum changes the last few bits — not an approximation of attention itself.
      </p>
      <p>
        The trick is the <em>online softmax</em>: when a later tile raises the running max, you rescale what you have
        already accumulated so the normalization stays exactly correct. In the update below,{' '}
        <MathDisplay inline latex={String.raw`\tilde\ell`} /> and <MathDisplay inline latex={String.raw`\tilde o`} />{' '}
        are <em>this</em> tile&apos;s own sum and weighted values measured against the updated max{' '}
        <MathDisplay inline latex={String.raw`m_i`} /> — so once the running totals are scaled by{' '}
        <MathDisplay inline latex={String.raw`e^{\,m_{i-1}-m_i}`} />, the new tile folds straight in with no extra
        factor.
      </p>
      <MathDisplay
        latex={String.raw`m_i = \max(m_{i-1}, \tilde m), \quad \ell_i = e^{\,m_{i-1}-m_i}\,\ell_{i-1} + \tilde\ell, \quad o_i = e^{\,m_{i-1}-m_i}\,o_{i-1} + \tilde o`}
      />
      <p>
        Those tiles live in the GPU&apos;s fast on-chip memory (SRAM), so the giant matrix never travels to slow memory
        at all. Same answer, a fraction of the memory traffic — the technique is &ldquo;IO-aware.&rdquo;
      </p>

      <h2>A short history</h2>
      <p>
        Flash Attention didn&apos;t arrive fully formed — it is the payoff of a chain of ideas about computing softmax
        without ever holding the whole row in memory. Step through the lineage:
      </p>

      <FlashAttentionTimeline />

      <p className="text-muted-foreground">
        The same idea spread fast beyond the original code: xformers&apos; <code>memory_efficient_attention</code>,
        OpenAI&apos;s Triton flash kernels, NVIDIA&apos;s cuDNN fused attention, and PyTorch&apos;s{' '}
        <code>scaled_dot_product_attention</code> (which dispatches to whichever is fastest). And PagedAttention (from
        vLLM) is a cousin, not a version — it is an attention kernel built around a paged KV-cache layout for
        memory-efficient serving, a different problem from FlashAttention&apos;s dense in-SRAM tiling. The fused kernel
        your in-browser Qwen runs during prefill is a <strong>FlashAttention-2-style</strong> tile kernel — v3&apos;s
        tricks are specific to NVIDIA&apos;s Hopper GPUs and FP8, so they don&apos;t apply here.
      </p>

      <h2>How each version actually works</h2>
      <p>
        The timeline above is the map; here is the territory. Each step fixed the previous one&apos;s bottleneck, so
        they make the most sense in order — one mechanism at a time.
      </p>

      <h3>The foundation: streaming softmax</h3>
      <p>
        Everything starts with one trick, so let&apos;s slow down and earn it. First, softmax itself, with real numbers.
        Softmax turns a row of scores into weights that sum to 1 by exponentiating each score and dividing by the total.
        Take the row <code>[2, 1, 3]</code>: the exponentials are{' '}
        <MathDisplay inline latex={String.raw`e^{2} \approx 7.39`} />,{' '}
        <MathDisplay inline latex={String.raw`e^{1} \approx 2.72`} />,{' '}
        <MathDisplay inline latex={String.raw`e^{3} \approx 20.09`} />; their sum is <code>30.19</code>; divide through
        and the weights are <code>0.245, 0.090, 0.665</code>. The biggest score wins most of the weight — that is the
        whole job.
      </p>
      <p>
        One practical wrinkle: <MathDisplay inline latex={String.raw`e^{x}`} /> explodes.{' '}
        <MathDisplay inline latex={String.raw`e^{89}`} /> already overflows a 32-bit float (the limit is about{' '}
        <MathDisplay inline latex={String.raw`e^{88.7}`} />
        ), and raw attention scores can get big. The standard fix is to subtract the row&apos;s max from every score
        before exponentiating — the largest term becomes <MathDisplay inline latex={String.raw`e^{0} = 1`} /> and
        nothing overflows. Dividing by the sum cancels the shift, so the weights come out identical. But notice what
        that fix costs: you need the <em>max of the whole row</em> before you can exponentiate <em>anything</em>. Hence
        two passes — one to find the max, one to sum the exponentials.
      </p>
      <p>
        The <strong>online softmax</strong> (2018) collapses that to a single pass: keep a <em>running</em> max and a{' '}
        <em>running</em> sum, and when a later value beats the max so far, <em>patch</em> the sum you already built by
        multiplying it by <MathDisplay inline latex={String.raw`e^{\,\text{old max}\; -\; \text{new max}}`} />. Why is
        that the exact right patch? Every term you accumulated has the form{' '}
        <MathDisplay inline latex={String.raw`e^{\,\text{score}\; -\; \text{old max}}`} />, and
      </p>
      <MathDisplay
        latex={String.raw`e^{\,s - m_{\text{old}}} \cdot e^{\,m_{\text{old}} - m_{\text{new}}} = e^{\,s - m_{\text{new}}}`}
      />
      <p>
        — one multiply re-expresses every old term against the new max, as if you had known it all along. Concretely, in
        the animation below the first chunk <code>[2, 1, 3]</code> finishes with max <code>3</code>; then chunk two
        contains a <code>5</code>, so the running sum and the running output are both multiplied by{' '}
        <MathDisplay inline latex={String.raw`e^{\,3 - 5} \approx 0.135`} /> before the new terms fold in. The 2021
        &ldquo;memory-efficient attention&rdquo; paper noticed you can run this over <em>chunks</em> of keys and values
        — so you never need the whole row in memory at once.
      </p>

      <StreamingSoftmaxDiagram />

      <h3>FlashAttention — tiling the whole thing into SRAM</h3>
      <p>
        A GPU has two kinds of memory: <strong>HBM</strong>, the big main memory that is far away and slow, and{' '}
        <strong>SRAM</strong>, a tiny scratchpad right next to the compute units that is roughly ten times faster. The
        naïve recipe builds the full N×N score matrix in HBM and drags it back and forth — the &ldquo;memory wall&rdquo;
        from the previous sub-chapter. <strong>FlashAttention</strong> (2022) keeps the scores out of HBM entirely. It
        cuts <code>Q</code>, <code>K</code>, and <code>V</code> into small tiles; for each tile of queries it streams
        over the key/value tiles, computes each little score tile <em>inside SRAM</em>, folds it straight into the
        running softmax, and throws it away. Only the finished output rows are written back. The giant score matrix
        never exists in slow memory — the same exact answer, a fraction of the traffic.
      </p>
      <p>
        How big is a tile? Just small enough that everything in flight fits on the desk. With a typical 64-wide head, a
        128-row query tile is <code>128 × 64</code> numbers ≈ 32 KB in fp32. A key tile and a value tile of the same
        shape are another 32 KB each, the little <code>128 × 128</code> score tile is 64 KB — and the running output{' '}
        <code>o</code> is itself a <code>128 × 64</code> tile, another 32 KB (only <code>m</code> and <code>ℓ</code> are
        single numbers per row). Add it up: ≈ 190 KB, right at the edge of the ~192 KB of SRAM sitting next to the
        compute units — which is exactly why the tiles are this size and not bigger. Production kernels squeeze further
        with 16-bit tiles and narrower key blocks, but the budget arithmetic — &ldquo;what fits in SRAM&rdquo; — is what
        picks the tile sizes.
      </p>
      <p className="text-muted-foreground">
        One honest footnote: the invariant v1 introduced is that the N×N score matrix never touches HBM. The exact{' '}
        <em>loop order</em> shown here — queries on the outside, keys/values streamed on the inside, each output row
        written once — is the cleaner schedule that <strong>v2</strong> settled on. The original v1 kernel actually
        looped the other way (keys/values outside, queries inside), revisiting the output as it went. The diagram below
        draws the modern order because that is what kernels run today.
      </p>

      <FlashTilingDiagram />

      <p>
        How much traffic does that actually save? At 4,096 tokens, the naïve recipe pushes the score matrix across the
        slow HBM bus four times — written as raw scores, read back for softmax, written as probabilities, read again to
        weight <code>V</code>. For this model&apos;s 8 heads in bf16 that is ≈ 1.07 GB of bus traffic{' '}
        <em>per full-attention layer</em>, for scores that are used once and thrown away. The tiled kernel moves exactly
        zero of those bytes. Drag the sequence length and watch the gap:
      </p>

      <FlashTrafficDiagram />

      <p>
        There is a second half to v1 that is easy to miss: <strong>training</strong>. The backward pass (computing
        gradients) normally needs the attention weights <em>again</em> — and storing the N×N matrix for later would
        defeat the entire point. FlashAttention&apos;s answer is <em>recomputation</em>: keep only the output and two
        small per-row statistics (that same max <code>m</code> and sum <code>ℓ</code>), then <em>re-derive</em> each
        score tile from Q and K inside SRAM when the backward pass needs it. That is deliberately doing <em>more</em>{' '}
        arithmetic to do <em>less</em> memory traffic — and it wins, because on a modern GPU the math is cheap and the
        trips to HBM are not. This counterintuitive trade is the heart of the whole &ldquo;IO-aware&rdquo; idea.
      </p>

      <h3>A 30-second tour of the GPU</h3>
      <p>
        The next two versions are about <em>filling the machine</em>, so you need a picture of the machine. A GPU is not
        one processor — it is more like an office park. An NVIDIA A100 has <strong>108 SMs</strong> (streaming
        multiprocessors): independent little processors, each with its own SRAM scratchpad (~192 KB) and its own{' '}
        <strong>Tensor Cores</strong>, the dedicated matrix-multiply units. Work arrives as <em>thread blocks</em> —
        each block is assigned to one SM, and inside a block the threads run in bundles of 32 called <em>warps</em>,
        which execute in lockstep. Two consequences matter here: an SM with no block assigned does <em>nothing</em>, and
        the kernel decides how many blocks exist. Zoom through the levels:
      </p>

      <GpuHierarchyDiagram />

      <h3>FlashAttention-2 — keep every core busy</h3>
      <p>
        FlashAttention-2 (2023) changes none of the math — it changes how the work is spread across the chip. Run the
        numbers with the office-park picture in mind. The original kernel launched roughly one thread block per (batch ×
        attention head): one prompt times 8 heads is <strong>8 blocks</strong> — on a 108-SM A100, about 7% of the chip
        working and 100 SMs sitting dark. v2&apos;s headline fix is to <em>also</em> split the query sequence into
        blocks: a 4,096-token prompt in 128-row tiles gives 32 query tiles, so 8 heads × 32 tiles ={' '}
        <strong>256 blocks</strong> — more than enough to light up every SM.
      </p>
      <p>
        It also re-partitions the work <em>inside</em> a block: instead of every warp computing a slice of each row and
        then combining results through shared memory, each warp owns a band of query rows outright — no cross-warp
        bookkeeping. And it shaves the non-matmul arithmetic, because regular math runs far slower than the Tensor
        Cores: the running output is kept <em>unnormalized</em> and divided by <code>ℓ</code> exactly once at the very
        end instead of being normalized at every step (the running max-rescale by{' '}
        <MathDisplay inline latex={String.raw`e^{\,m_{i-1}-m_i}`} /> still happens each tile — it is only the{' '}
        <code>ℓ</code>-division that is deferred), and for causal masks (where future tokens cannot be attended anyway)
        whole score tiles that would be fully masked are simply never computed — measured at ~1.7–1.8× by itself.
        Together: about twice as fast as v1, with the attention kernel hitting up to 73% of the A100&apos;s theoretical
        FP16 peak. (That occupancy idea is exactly what drives the decode story below.)
      </p>

      <Flash2ParallelismDiagram />

      <h3>FlashAttention-3 — overlap and low precision</h3>
      <p>
        FlashAttention-3 (2024) is tuned for NVIDIA&apos;s Hopper GPUs (the H100), and it starts from a wild imbalance
        in the hardware. The H100&apos;s Tensor Cores deliver about <strong>989 TFLOPS</strong> of fp16 matrix multiply
        — but the little units that compute exponentials (which every softmax needs) manage only about{' '}
        <strong>3.9 TFLOPS</strong>. That is a <strong>256×</strong> speed gap between the two kinds of math attention
        alternates between. Run matmul → softmax → matmul strictly in order, and the most expensive silicon on the chip
        spends much of its time waiting for the cheap part to finish.
      </p>
      <p>
        FlashAttention-3&apos;s answer is to stop taking turns. Hopper lets warps <em>specialize</em>:{' '}
        <strong>producer</strong> warps do nothing but tell the <strong>TMA</strong> (a dedicated copy engine) to fetch
        the next tiles from HBM, while <strong>consumer</strong> warpgroups do nothing but math. Then two consumer
        warpgroups play <em>pingpong</em>: while one runs its softmax on the exp units, the other&apos;s matmuls keep
        the Tensor Cores fed — and they swap. Each tile is really two matmuls — the scores <code>QKᵀ</code>, then the
        value mix <code>P·V</code> — with the softmax wedged between, and it is those that get overlapped:
      </p>

      <Flash3AsyncDiagram />

      <p>
        The second lever is <strong>FP8</strong> — and it deserves a slower look, because &ldquo;8-bit float&rdquo;
        sounds innocent until you see what it costs. An fp8 number (the E4M3 format) has 256 possible bit patterns and a
        maximum value of 448; matmuls in fp8 run about twice as fast as fp16. The catch: with so few representable
        values, everything hinges on the <em>scale factor</em> that maps your data onto them. Real attention inputs
        contain occasional <em>outliers</em> — one huge value in a sea of small ones. Pick one scale for the whole
        tensor and that single outlier stretches the grid so far that all the small values collapse onto a handful of
        levels and their information is gone. FlashAttention-3 rescues fp8 twice: <em>block quantization</em> gives each
        small tile its own scale, so one outlier only coarsens its own block — and <em>incoherent processing</em>{' '}
        multiplies Q and K by a fixed random rotation that smears the outlier&apos;s energy across every dimension
        before rounding. Before any rounding that rotation is mathematically free — rotate both sides and the dot
        products are unchanged. After fp8 rounding nothing is exactly preserved; the rotation&apos;s job is to make the
        rounding <em>cheap</em>, and the error it saves is what the widget below measures:
      </p>

      <Fp8QuantDiagram />

      <p>
        And the story keeps going: <strong>FlashAttention-4</strong> (2026) carries the same invariant onto
        NVIDIA&apos;s next generation, the Blackwell B200 — where matmul throughput roughly doubled again while the rest
        of the chip didn&apos;t — reaching up to 1,613 TFLOPS (71% utilization). Our deep-dive stops at v3 because
        v4&apos;s tricks are Blackwell-specific, but notice the pattern across every version: the{' '}
        <em>algorithm&apos;s invariant never changes</em> — the N×N matrix never touches slow memory — and each release
        just re-tunes that invariant to whatever the newest silicon&apos;s bottleneck happens to be.
      </p>

      <h2>Does your in-browser Qwen use it?</h2>
      <p>
        Partly — and the honest answer is the interesting one. The WebGPU backend running this model <em>has</em> a
        fused, tiled flash-attention kernel with the online softmax above, and it runs during <em>prefill</em> of the 6
        full-attention layers. But the token-by-token <em>decode</em> you watch in the demo deliberately does{' '}
        <em>not</em> use it: at 0.8B with 8 query heads the fused kernel would launch only 8 GPU workgroups and leave
        most of the chip idle, so an occupancy gate routes decode back to the plain three-step path (matmul → softmax →
        matmul), which fans out far more work and measured ~90% faster here. And only 6 of 24 layers run softmax
        attention at all — the other 18 are GatedDeltaNet. So: flash during prefill, plain during decode, on purpose.
      </p>
      <p>Toggle between the two and watch the dispatch route the call to the kernel that actually runs:</p>

      <NativeAttentionDiagram />
    </Prose>
  );
}

/** Sub-chapter 4.3 — turning "memory-bound" from a word into a number: the roofline. */
export function AttentionRooflineSection() {
  return (
    <Prose>
      <h1>The roofline</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 4, Self-attention — putting an actual number on &ldquo;memory-bound.&rdquo;
      </p>
      <p>
        The <a href="/chapters/attention/memory-wall">memory wall</a> kept leaning on one word — <em>memory-bound</em> —
        to explain why attention, and single-token decode in general, spends its time waiting on memory instead of
        computing. That was a story. The <strong>roofline</strong> turns it into a number you can plot, and that number
        tells you exactly when the story stops being true.
      </p>

      <h2>Two ceilings, and which one you hit first</h2>
      <p>
        A GPU can run out of road for two completely independent reasons, and whichever you reach first becomes your
        ceiling. One is <strong>compute</strong> — how many math operations per second the Tensor Cores can do (an H100
        in fp16 tops out near <strong>989 TFLOPS</strong>). The other is <strong>memory bandwidth</strong> — how many
        bytes per second you can stream out of HBM (the same H100: <strong>3.35 TB/s</strong>). One caps the math; the
        other caps the bytes moved. A faster engine doesn&apos;t help if you&apos;re starved on fuel delivery.
      </p>
      <p>
        Which ceiling binds you is a property of the <em>workload</em>, not the chip: its{' '}
        <strong>arithmetic intensity</strong> — the FLOPs of work done for every byte moved, measured in{' '}
        <code>ops:byte</code>.
      </p>
      <MathDisplay
        latex={String.raw`\text{intensity} = \frac{\text{work (FLOPs)}}{\text{memory traffic (bytes)}}, \qquad \text{ridge} = \frac{\text{peak compute}}{\text{bandwidth}} = \frac{989\ \text{TFLOP/s}}{3.35\ \text{TB/s}} \approx 295\ \text{ops:byte}`}
      />
      <p>
        That <strong>ridge</strong> at ~295 ops:byte is the break-even point. Do <em>fewer</em> than 295 operations per
        byte and you finish the math long before the next bytes arrive — you are <strong>memory-bound</strong>, idling
        on the left of the ridge. Do <em>more</em> and the bytes arrive faster than you can crunch them — you are{' '}
        <strong>compute-bound</strong>, on the right. The roofline plots achievable performance against intensity: a
        diagonal bandwidth ceiling that climbs until it meets the flat compute ceiling exactly at the ridge.
      </p>

      <h2>Where prefill and decode land</h2>
      <p>
        The two halves of inference sit on opposite sides. <strong>Prefill</strong> processes prompt positions in
        parallel — there is no left-to-right dependency between them, unlike decode — so every weight it reads is reused
        across <em>hundreds</em> of positions at a time: a mountain of matmul per byte. (Whether that arrives as one
        pass or several is a scheduling choice; see <a href="/chapters/kv-cache/batching">chunked prefill</a>. It stays
        compute-bound either way.) It sits comfortably right of the ridge, <strong>compute-bound</strong>.{' '}
        <strong>Decode</strong> at batch 1 is the opposite extreme: to produce <em>one</em> token it streams the entire
        model&apos;s weights through the chip exactly once, doing almost no math per byte. It falls off the far-left
        cliff, <strong>deeply memory-bound</strong> — the same idle-waiting-on-memory the rest of this course keeps
        running into.
      </p>
      <p>
        Drag the batch slider and watch the decode dot march. Batching <code>N</code> requests lets all <code>N</code>{' '}
        reuse each weight read, so intensity climbs roughly <code>×N</code> and the dot slides right up the diagonal
        toward the ridge — flipping from memory-bound to compute-bound only once there&apos;s finally enough math per
        byte to keep the FLOPs busy:
      </p>

      <RooflineDiagram />

      <p>
        So &ldquo;memory-bound&rdquo; was never a fixed property of the model — it&apos;s a property of{' '}
        <em>how you run it</em>. One user waiting on one token sits on the cliff; a server batching hundreds of users
        climbs the diagonal until compute finally becomes the thing worth optimizing. That climb is the whole subject of
        the <a href="/chapters/kv-cache/batching">batching</a> sub-chapter later on — the roofline is the map it reads.
      </p>
    </Prose>
  );
}
