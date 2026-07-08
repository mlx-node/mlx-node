// Sub-chapter 13.1 — Pretraining data curation.
//
// A "go deeper" deep-dive hung off chapter 13 (Training & teacher forcing).
// Pure, model-free reading prose — SSR-safe, so the prerenderer renders it
// straight to crawlable static HTML (see scripts/prerender.entry.tsx). Wrapped
// in <Prose> by the section route / prerenderer, so it returns just the
// article content.
//
// Chapter 13 currently compresses pretraining data into one clause —
// "trillions of tokens of generic web text, books, and code" — with zero
// curation detail. This sub-chapter unpacks that clause into three
// well-documented pillars, each grounded in a primary source:
//   - Deduplication: Lee et al. 2021 (arXiv:2107.06499) — exact-substring
//     (suffix-array) dedup + MinHash/LSH near-duplicate dedup; raw web corpora
//     are 0.39%-13.63% near-duplicate by document, and training on the
//     undeduplicated version makes models emit memorized text ~10x more often.
//   - Quality filtering: classifier-based (GPT-3's logistic-regression
//     classifier, arXiv:2005.14165; FineWeb-Edu's Llama-3-70B-distilled
//     educational-value scorer) and perplexity-based (CCNet's per-language
//     KenLM-on-Wikipedia scoring — confirmed only via secondary sources here,
//     flagged as such, not presented as arXiv-verified).
//   - Sourcing/mixture: Common Crawl as the near-universal raw base; FineWeb
//     (Penedo et al. 2024, arXiv:2406.17557) as a fully documented CC-to-15T
//     pipeline, plus LLaMA's published non-uniform mixture ratios
//     (arXiv:2302.13971) as a concrete example of deliberately blending web
//     text with books/code/academic sources.
//
// Honesty close: the Qwen3 Technical Report (arXiv:2505.09388) discloses
// ~36T tokens across 3 stages and broad category labels, but names no
// dedup/filtering method and no source-percentage breakdown. No dedicated
// Qwen3.5 / Qwen3-Next (plain text/hybrid) technical report could be found as
// of 2026-07-01 — only the separate, multimodal Qwen3.5-Omni report
// (arXiv:2604.15804), equally silent on curation. Stanford CRFM's Foundation
// Model Transparency Index (Dec 2025) independently scores Alibaba/Qwen at 0
// on data-acquisition methods, sources, crawler/opt-out handling, and
// composition percentages — the same non-disclosure norm as most other major
// labs.

import * as React from 'react';

import { Prose } from '../Prose';
import { MathDisplay } from '../scaffolding/MathDisplay';
import { DataCurationFunnel } from '../widgets/DataCurationFunnel';

/** Sub-chapter 13.1 — what "trillions of tokens of generic web text" actually took to build. */
export function TrainingDataCurationSection() {
  return (
    <Prose>
      <h1>Pretraining data curation</h1>
      <p className="text-muted-foreground">
        Go deeper · Chapter 13, Training &amp; teacher forcing — the last chapter said &ldquo;trillions of tokens of
        generic web text, books, and code.&rdquo; Here is what actually goes into that clause.
      </p>
      <p>
        The training objective from the last chapter — minimize <code>-log p(next token)</code> at every position — is
        the same one line for every modern LLM. What differs, and what quietly does most of the work, is{' '}
        <em>what text you feed it</em>. Scrape the raw web and train on it as-is and you get a worse model than if you
        first clean it — not because the objective changes, but because raw web text is full of problems the objective
        cannot see past on its own: it is riddled with near-duplicates, much of it is boilerplate or spam, and it is
        not sourced or mixed the way a curated corpus is. This sub-chapter unpacks the three pillars behind that one
        glossed-over clause: <strong>deduplication</strong>, <strong>quality filtering</strong>, and{' '}
        <strong>sourcing &amp; mixture</strong>.
      </p>

      <h2>Deduplication: the web repeats itself more than you&apos;d think</h2>
      <p>
        Common Crawl and similar web scrapes are not a clean sample of unique documents. The same article gets
        syndicated across dozens of news mirrors, the same boilerplate legal notice appears on thousands of unrelated
        sites, and templated pages differ by only a sentence. Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, and
        Carlini measured exactly how much of this survives into standard training corpora (<em>Deduplicating Training
        Data Makes Language Models Better</em>, 2021): <strong>3.04%</strong> of C4, <strong>13.63%</strong> of
        RealNews, <strong>4.86%</strong> of LM1B, and <strong>0.39%</strong> of Wiki-40B are near-duplicate documents.
        In one extreme case, a single 61-word sentence in C4 was found repeated <strong>over 60,000 times</strong>.
      </p>
      <p>
        Why bother removing it? Because a model trained on heavily-duplicated text doesn&apos;t just waste compute
        re-learning the same sentence — it starts to <em>memorize</em> it. Lee et al. found that deduplication cut how
        often a trained model emits verbatim memorized training text by roughly <strong>10×</strong>, while also
        shrinking the dataset by up to 19% and reducing accidental train/test overlap (which had been inflating
        benchmark scores on over 4% of standard validation sets).
      </p>
      <p>
        The paper uses two complementary techniques. <strong>Exact-substring deduplication</strong> builds a suffix
        array over the entire corpus and removes any verbatim span of 50+ BPE tokens that occurs in more than one
        document — catches copy-pasted boilerplate and quoted blocks. <strong>Near-duplicate deduplication</strong>{' '}
        (&ldquo;NearDup&rdquo;) is the harder problem: two documents that are <em>almost</em> the same but not
        byte-identical (a reprinted article with a different headline, a templated page with one field changed). Lee et
        al.&apos;s tool is <strong>MinHash</strong>: represent each document as a set of overlapping 5-grams, hash that
        set into a compact signature (here, 9,000 hash values split into 20 bands of 450), and flag two documents as
        near-duplicates when enough of their signature matches — corresponding to a Jaccard similarity around{' '}
        <MathDisplay inline latex={String.raw`J(A,B) = \frac{|A \cap B|}{|A \cup B|} \gtrsim 0.8`} />. MinHash-based
        LSH (locality-sensitive hashing) is now the industry-standard way to deduplicate web-scale text — reused, with
        its own tuned parameters, by essentially every modern open pipeline, FineWeb included.
      </p>

      <h2>Quality filtering: two classic recipes</h2>
      <p>
        Deduplication removes <em>repeated</em> text; quality filtering removes text that is unique but{' '}
        <em>bad</em> — link farms, keyword-stuffed spam, auto-generated boilerplate, pages that are mostly navigation
        menus. Two approaches have dominated for years, and both are still in active use today.
      </p>
      <p>
        <strong>Classifier-based filtering</strong> trains a small model to distinguish &ldquo;text that looks like the
        good stuff&rdquo; from raw web crawl. GPT-3&apos;s paper describes the canonical version: a logistic-regression
        classifier trained with curated corpora (WebText, Wikipedia, a books corpus) as positive examples against raw
        Common Crawl as negative examples, then used to resample Common Crawl toward documents that score more
        Wikipedia/book-like. The same idea resurfaces two years later as <strong>FineWeb-Edu</strong>: Llama-3-70B-Instruct
        was prompted to rate 500k web pages 0-5 for educational value, and a lightweight classifier was distilled from
        those ratings to score FineWeb&apos;s full 15T-token corpus at scale — producing a 1.3T-token subset that
        substantially outperforms the unfiltered corpus on knowledge-heavy benchmarks like MMLU.
      </p>
      <p>
        <strong>Perplexity-based filtering</strong> takes a different, cheaper approach: score every paragraph by how
        &ldquo;surprising&rdquo; it looks to a small language model trained on known-good text, and discard the
        high-perplexity (surprising, likely boilerplate-or-spam) tail. CCNet is the widely-cited example — a 5-gram
        KenLM model trained per-language on that language&apos;s Wikipedia, used to score crawled paragraphs and keep
        the low-perplexity, Wikipedia-like ones, later reused as a filtering stage in RedPajama and Dolma. (This
        project could only confirm CCNet&apos;s method via secondary sources describing the paper, not the original
        paper&apos;s text directly — worth flagging, since every other claim on this page traces to a primary source we
        fetched ourselves.)
      </p>

      <h2>Sourcing &amp; mixture: Common Crawl is the base, not the whole recipe</h2>
      <p>
        Nearly every open pretraining corpus starts from the same place: <strong>Common Crawl</strong>, a public,
        continuously-updated scrape of the web, released as monthly snapshots. What separates a raw Common Crawl dump
        from a usable pretraining dataset is exactly the pipeline above — and{' '}
        <strong>FineWeb</strong> (Penedo et al., 2024) is the cleanest fully-documented worked example: URL blocklist
        filtering, text extraction (via the <code>trafilatura</code> library), fastText language identification,
        Gopher/C4-style plus custom heuristic quality filters, and finally MinHash near-duplicate dedup — turning 96
        raw Common Crawl snapshots into a 15-trillion-token curated English dataset. Toggle to the &ldquo;Mixture&rdquo;
        view below to see the second half of the sourcing story: labs don&apos;t train on web text alone, they{' '}
        <em>deliberately blend</em> it with books, code, and reference text in fixed, non-uniform ratios — LLaMA&apos;s
        published mixture table is the clearest cited example, up-sampling scarce high-quality sources (Wikipedia,
        books, ArXiv) across multiple epochs while running the much larger Common Crawl slice for barely more than one.
      </p>

      <DataCurationFunnel />

      <p className="text-muted-foreground">
        Read the pipeline view as a taper, not a spreadsheet: only the entry (96 snapshots) and the two exits (15T
        tokens for FineWeb, 1.3T for the classifier-filtered FineWeb-Edu subset) are numbers FineWeb actually publishes.
        The dedup statistics underneath it, by contrast, are real measurements from Lee et al. — shown on their own row
        so the illustrative taper and the measured facts are never mistaken for each other.
      </p>

      <h2>Does your in-browser Qwen tell you how its training data was cleaned?</h2>
      <p>
        No — and this time the honest answer is less about this specific browser demo and more about the entire
        industry it comes from. The <strong>Qwen3 Technical Report</strong> does disclose real numbers about scale:{' '}
        <strong>≈36 trillion tokens</strong> across three pretraining stages (Stage 1, general, <code>&gt;30T</code>{' '}
        tokens; Stage 2, reasoning/knowledge boost, <code>+≈5T</code> more; Stage 3, long-context, hundreds of billions
        more at 32,768-token sequences), plus broad category labels — STEM, code, books, multilingual text, and
        synthetic data extracted and generated with the help of Qwen2.5-VL (for OCR&apos;ing PDF-like documents),
        Qwen2.5-Math, and Qwen2.5-Coder. What it does <strong>not</strong> disclose is any of the machinery on this
        page: no deduplication method or tooling is named, no quality-filtering methodology (classifier or perplexity)
        is described, and there is no percentage breakdown of web-vs-books-vs-code-vs-academic composition — nothing
        resembling the LLaMA table above.
      </p>
      <p>
        It gets murkier still for the exact model this course runs. As of 2026-07-01, no dedicated{' '}
        <em>Qwen3.5</em> or <em>Qwen3-Next</em> technical report — covering the plain hybrid text/GatedDeltaNet line
        this browser tab actually runs, as opposed to the separate multimodal line — could be found on arXiv. Two 2026
        Qwen papers exist, and neither closes that gap: the <em>Qwen3.5-Omni Technical Report</em> covers an
        architecturally different omnimodal (text+vision+audio) model and discloses data scale (100M+ hours of
        audio-visual data) but, again, zero curation methodology; the earlier{' '}
        <em>Qwen3-Coder-Next Technical Report</em> (arXiv:2603.00729, Feb 2026) builds on a Qwen3-Next base but is
        scoped to agentic coding capability — mid-training and RL on top of an existing base, not a pretraining-data
        disclosure. Neither is a guess dressed up as a fact: this is the honest result of actually searching and
        coming up empty.
      </p>
      <p className="text-muted-foreground">
        An independent audit backs this up. Stanford CRFM&apos;s Foundation Model Transparency Index (December 2025
        company report) scores Alibaba/Qwen at <strong>0 — not disclosed</strong> — on methods for acquiring
        pretraining data, public dataset sources used, crawler identity and robots.txt opt-out handling, licensed-data
        specifics, and language/domain composition percentages. That is not a Qwen-specific failing: OpenAI,
        Anthropic, Google, and Meta withhold the same source-level detail for their own frontier releases. So the
        honest picture is this: everything on this page above the fold — dedup, quality filtering, Common Crawl,
        FineWeb, LLaMA&apos;s mixture — is real, documented, and verifiable, because it comes from labs and papers that
        chose to publish it. The recipe actually used to clean the training corpus behind the weights running in your
        browser right now — undisclosed for this exact checkpoint, though likely a similar order of magnitude to the
        ~36 trillion tokens Qwen3 discloses — is, like almost every other frontier model&apos;s, proprietary.
      </p>
    </Prose>
  );
}
